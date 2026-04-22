// Copyright © SixtyFPS GmbH <info@slint.dev>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

use i_slint_core::api::{GraphicsAPI, PhysicalSize as PhysicalWindowSize, Window};
use i_slint_core::graphics::RequestedGraphicsAPI;
use i_slint_core::partial_renderer::DirtyRegion;
use i_slint_core::platform::PlatformError;

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use wgpu_28 as wgpu;

use crate::SkiaSharedContext;

#[cfg(target_family = "windows")]
mod dx12;
#[cfg(target_vendor = "apple")]
mod metal;
#[cfg(all(target_family = "unix", not(target_vendor = "apple")))]
mod vulkan;

// ---------------------------------------------------------------------------
// Viewport blit: renders external wgpu textures directly to the swapchain
// surface, bypassing Slint's Image element and Skia renderer entirely.
// ---------------------------------------------------------------------------

/// A viewport region to blit onto the swapchain surface.
pub struct ViewportBlit {
    /// The external texture to sample from.
    pub texture: wgpu::Texture,
    /// Viewport rectangle in physical pixels (x, y, width, height).
    pub rect: [f32; 4],
}

/// WGSL shader for viewport blit: renders a textured quad at a given position.
const BLIT_SHADER: &str = r#"
struct BlitUniforms {
    // Viewport rect in NDC: (x, y, width, height) mapped to [-1,1]
    rect: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: BlitUniforms;
@group(0) @binding(1) var src_texture: texture_2d<f32>;
@group(0) @binding(2) var src_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // Full-screen quad vertices: 0,1,2,3 → two triangles via triangle-strip
    let x = f32(vi & 1u);
    let y = f32(vi >> 1u);
    var out: VertexOutput;
    // Map [0,1] quad to the viewport rect in NDC
    out.position = vec4<f32>(
        u.rect.x + x * u.rect.z,
        u.rect.y + y * u.rect.w,
        0.0, 1.0
    );
    out.uv = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(src_texture, src_sampler, in.uv);
}
"#;

/// Lazily-initialized blit pipeline resources.
struct BlitPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
}

impl BlitPipeline {
    fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Viewport Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Viewport Blit BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Viewport Blit Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Viewport Blit Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Viewport Blit Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Uniform buffer for one viewport rect (16 bytes = vec4<f32>)
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Viewport Blit Uniforms"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { pipeline, bind_group_layout, sampler, uniform_buffer }
    }
}

// ---------------------------------------------------------------------------

/// This surface renders into the given window using Metal. The provided display argument
/// is ignored, as it has no meaning on macOS.
pub struct WGPUSurface {
    gr_context: RefCell<skia_safe::gpu::DirectContext>,
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: RefCell<wgpu::SurfaceConfiguration>,
    surface: wgpu::Surface<'static>,
    textures_to_transition_for_sampling: RefCell<Vec<wgpu::Texture>>,
    texture_image_cache: RefCell<HashMap<wgpu::Texture, skia_safe::Image>>,
    /// External textures to blit onto the swapchain after Skia rendering.
    viewport_blits: RefCell<Vec<ViewportBlit>>,
    /// Lazily-initialized blit pipeline.
    blit_pipeline: RefCell<Option<BlitPipeline>>,
    /// Temporary reference to the current swapchain frame view, set during render().
    current_frame_view: RefCell<Option<wgpu::TextureView>>,
    backend: Backend,
}

impl super::Surface for WGPUSurface {
    fn new(
        _shared_context: &SkiaSharedContext,
        window_handle: Arc<dyn raw_window_handle::HasWindowHandle + Send + Sync>,
        display_handle: Arc<dyn raw_window_handle::HasDisplayHandle + Send + Sync>,
        size: PhysicalWindowSize,
        requested_graphics_api: Option<RequestedGraphicsAPI>,
    ) -> Result<Self, PlatformError> {
        let (instance, adapter, device, queue, surface) =
            i_slint_core::graphics::wgpu_28::init_instance_adapter_device_queue_surface(
                Box::new(WindowAndDisplayHandle(window_handle, display_handle)),
                requested_graphics_api,
                wgpu::Backends::GL /* we're not mapping that to skia because we can't save/restore state */
                    .union(if cfg!(target_os = "windows") {
                        wgpu::Backends::VULKAN
                    } else {
                        wgpu::Backends::empty()
                    }),
            )?;

        let mut surface_config =
            surface.get_default_config(&adapter, size.width, size.height).unwrap();

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities
            .formats
            .iter()
            .find(|f| {
                matches!(f, wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm)
            })
            .copied()
            .unwrap_or_else(|| swapchain_capabilities.formats[0]);
        surface_config.format = swapchain_format;
        surface.configure(&device, &surface_config);

        let backend: Backend = adapter.get_info().backend.try_into()?;

        let gr_context = backend.make_context(&adapter, &device, &queue);

        Ok(Self {
            gr_context: RefCell::new(
                gr_context.ok_or_else(|| {
                    PlatformError::from("Failed to create Skia context from WGPU")
                })?,
            ),
            instance,
            device,
            queue,
            surface_config: surface_config.into(),
            surface,
            textures_to_transition_for_sampling: RefCell::new(Vec::new()),
            texture_image_cache: RefCell::new(HashMap::new()),
            viewport_blits: RefCell::new(Vec::new()),
            blit_pipeline: RefCell::new(None),
            current_frame_view: RefCell::new(None),
            backend,
        })
    }

    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn resize_event(&self, size: PhysicalWindowSize) -> Result<(), PlatformError> {
        {
            let gr_context = &mut self.gr_context.borrow_mut();
            // This is brute force, but for the lack of access to the fences this seems to work: Avoid any pending work so that
            // IDXGISwapChain::ResizeBuffers doesn't complain that the surface is still in use.
            gr_context.flush_submit_and_sync_cpu();
        }

        let mut surface_config = self.surface_config.borrow_mut();

        // Prefer FIFO modes over possible Mailbox setting for frame pacing and better energy efficiency.
        surface_config.present_mode = wgpu::PresentMode::AutoVsync;
        surface_config.width = size.width;
        surface_config.height = size.height;

        self.surface.configure(&self.device, &surface_config);
        self.texture_image_cache.borrow_mut().clear();
        Ok(())
    }

    fn render(
        &self,
        _window: &Window,
        size: PhysicalWindowSize,
        callback: &dyn Fn(
            &skia_safe::Canvas,
            Option<&mut skia_safe::gpu::DirectContext>,
            u8,
        ) -> Option<DirtyRegion>,
        pre_present_callback: &RefCell<Option<Box<dyn FnMut()>>>,
    ) -> Result<(), PlatformError> {
        let gr_context = &mut self.gr_context.borrow_mut();

        let frame = match self.surface.get_current_texture() {
            Ok(texture) => texture,
            Err(wgpu::SurfaceError::Timeout) => {
                self.surface.get_current_texture().map_err(|e| {
                    format!("Error obtaining current surface texture after timeout: {e}")
                })?
            }
            // Outdated or lost: re-configure and try again
            Err(_) => {
                self.surface.configure(&self.device, &*self.surface_config.borrow());
                self.surface.get_current_texture().map_err(|e| {
                    format!("Error obtaining current surface texture after initial error: {e}")
                })?
            }
        };
        // Store the frame view for viewport blits (used by execute_viewport_blits
        // which is called from inside the callback via render_components_to_canvas).
        *self.current_frame_view.borrow_mut() = Some(
            frame.texture.create_view(&wgpu::TextureViewDescriptor::default())
        );

        let skia_surface = self.backend.make_surface(size, gr_context, &frame);

        let mut skia_surface = skia_surface
            .ok_or_else(|| PlatformError::from("Failed to create Skia surface from WGPU"))?;

        // wgpu doesn't expose EGL_EXT_buffer_age, so assume triple buffering
        // (worst-case for FIFO/AutoVsync). This enables partial rendering to
        // union the last 2 frames' dirty regions instead of repainting everything.
        callback(skia_surface.canvas(), Some(gr_context), 3);

        let textures_to_transition = self.textures_to_transition_for_sampling.take();
        if !textures_to_transition.is_empty() {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Skia texture transition encoder"),
            });
            encoder.transition_resources(
                std::iter::empty(),
                textures_to_transition.iter().map(|texture| wgpu::TextureTransition {
                    texture,
                    selector: None,
                    state: wgpu::TextureUses::RESOURCE,
                }),
            );

            self.queue.submit(Some(encoder.finish()));
        }

        gr_context.submit(None);

        // Clear temporary frame view reference
        *self.current_frame_view.borrow_mut() = None;

        if let Some(pre_present_callback) = pre_present_callback.borrow_mut().as_mut() {
            pre_present_callback();
        }

        frame.present();

        Ok(())
    }

    fn bits_per_pixel(&self) -> Result<u8, PlatformError> {
        Ok(match self.surface_config.borrow().format {
            wgpu_28::TextureFormat::Rgba8Unorm
            | wgpu_28::TextureFormat::Rgba8UnormSrgb
            | wgpu_28::TextureFormat::Bgra8Unorm
            | wgpu_28::TextureFormat::Bgra8UnormSrgb => 32,
            fmt @ _ => return Err(format!("Unsupported surface format {:#?}", fmt).into()),
        })
    }

    fn with_graphics_api(&self, callback: &mut dyn FnMut(GraphicsAPI<'_>)) {
        let api = i_slint_core::graphics::create_graphics_api_wgpu_28(
            self.instance.clone(),
            self.device.clone(),
            self.queue.clone(),
        );
        callback(api)
    }

    fn import_wgpu_texture(
        &self,
        canvas: &skia_safe::Canvas,
        any_wgpu_texture: &i_slint_core::graphics::WGPUTexture,
    ) -> Option<skia_safe::Image> {
        let texture = match any_wgpu_texture {
            #[cfg(feature = "unstable-wgpu-27")]
            i_slint_core::graphics::WGPUTexture::WGPU27Texture(..) => return None,
            #[cfg(feature = "unstable-wgpu-28")]
            i_slint_core::graphics::WGPUTexture::WGPU28Texture(texture) => texture.clone(),
        };

        // Skia won't submit commands right away, so remember the texture and transition before
        // submitting.
        self.textures_to_transition_for_sampling.borrow_mut().push(texture.clone());

        // Cache the Skia Image wrapper — import_texture is expensive (creates new
        // Skia backend texture + Image each call) but the underlying wgpu::Texture
        // handles don't change between frames (only on resize/reallocation).
        let cache = self.texture_image_cache.borrow();
        if let Some(cached) = cache.get(&texture) {
            return Some(cached.clone());
        }
        drop(cache);

        let image = self.backend.import_texture(canvas, texture.clone())?;
        self.texture_image_cache.borrow_mut().insert(texture, image.clone());
        Some(image)
    }

    fn set_viewport_blits(&self, blits: Vec<ViewportBlit>) {
        *self.viewport_blits.borrow_mut() = blits;
    }

    fn has_viewport_blits(&self) -> bool {
        !self.viewport_blits.borrow().is_empty()
    }

    /// Blit-only render: acquire swapchain, blit viewports onto the existing
    /// buffer content (which retains valid UI from the last full paint), present.
    /// No Skia, no component tree, no dirty evaluation.
    fn render_blits_only(&self) -> Result<(), PlatformError> {
        if self.viewport_blits.borrow().is_empty() {
            return Ok(());
        }

        let gr_context = &mut self.gr_context.borrow_mut();
        let frame = match self.surface.get_current_texture() {
            Ok(texture) => texture,
            Err(wgpu::SurfaceError::Timeout) => {
                self.surface.get_current_texture().map_err(|e| {
                    format!("Error obtaining surface texture for blit-only: {e}")
                })?
            }
            Err(_) => {
                self.surface.configure(&self.device, &*self.surface_config.borrow());
                self.surface.get_current_texture().map_err(|e| {
                    format!("Error obtaining surface texture for blit-only after reconfig: {e}")
                })?
            }
        };

        *self.current_frame_view.borrow_mut() = Some(
            frame.texture.create_view(&wgpu::TextureViewDescriptor::default())
        );

        self.execute_viewport_blits();

        *self.current_frame_view.borrow_mut() = None;
        gr_context.submit(None);
        frame.present();

        Ok(())
    }

    fn execute_viewport_blits(&self) {
        let blits = self.viewport_blits.borrow();
        let frame_view_ref = self.current_frame_view.borrow();
        let Some(frame_view) = frame_view_ref.as_ref() else { return };
        if blits.is_empty() { return; }

        let surface_format = self.surface_config.borrow().format;
        let mut bp = self.blit_pipeline.borrow_mut();
        let pipeline = bp.get_or_insert_with(|| BlitPipeline::new(&self.device, surface_format));

        let surface_w = self.surface_config.borrow().width as f32;
        let surface_h = self.surface_config.borrow().height as f32;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Viewport Blit Encoder"),
        });

        for blit in blits.iter() {
            // Convert pixel rect to NDC: x,y → [-1,1], y flipped
            let ndc_x = blit.rect[0] / surface_w * 2.0 - 1.0;
            let ndc_y = 1.0 - (blit.rect[1] + blit.rect[3]) / surface_h * 2.0;
            let ndc_w = blit.rect[2] / surface_w * 2.0;
            let ndc_h = blit.rect[3] / surface_h * 2.0;

            let uniform_data: [f32; 4] = [ndc_x, ndc_y, ndc_w, ndc_h];
            self.queue.write_buffer(
                &pipeline.uniform_buffer,
                0,
                bytemuck::cast_slice(&uniform_data),
            );

            let tex_view = blit.texture.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Viewport Blit BG"),
                layout: &pipeline.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pipeline.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&tex_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&pipeline.sampler),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Viewport Blit Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..4, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
    }
}

struct WindowAndDisplayHandle(
    Arc<dyn raw_window_handle::HasWindowHandle + Send + Sync>,
    Arc<dyn raw_window_handle::HasDisplayHandle + Send + Sync>,
);

impl raw_window_handle::HasWindowHandle for WindowAndDisplayHandle {
    fn window_handle(
        &self,
    ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        self.0.window_handle()
    }
}

impl raw_window_handle::HasDisplayHandle for WindowAndDisplayHandle {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        self.1.display_handle()
    }
}

enum Backend {
    #[cfg(target_vendor = "apple")]
    Metal,
    #[cfg(target_family = "windows")]
    Dx12,
    #[cfg(all(target_family = "unix", not(target_vendor = "apple")))]
    Vulkan,
}

impl TryFrom<wgpu::Backend> for Backend {
    type Error = PlatformError;

    fn try_from(wgpu_backend: wgpu::Backend) -> Result<Self, Self::Error> {
        match wgpu_backend {
            wgpu_28::Backend::Noop => {
                Err(PlatformError::from("Cannot use WGPU Noop backend with Skia"))
            }
            #[cfg(all(target_family = "unix", not(target_vendor = "apple")))]
            wgpu_28::Backend::Vulkan => Ok(Self::Vulkan),
            #[cfg(target_vendor = "apple")]
            wgpu_28::Backend::Metal => Ok(Self::Metal),
            #[cfg(target_family = "windows")]
            wgpu_28::Backend::Dx12 => Ok(Self::Dx12),
            other @ _ => Err(PlatformError::from(format!(
                "Unsupported WGPU backend for use with Skia: {}",
                other.to_string()
            ))),
        }
    }
}

impl Backend {
    fn make_context(
        &self,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Option<skia_safe::gpu::DirectContext> {
        match self {
            #[cfg(target_vendor = "apple")]
            Self::Metal => metal::make_metal_context(device, queue),
            #[cfg(target_family = "windows")]
            Self::Dx12 => unsafe { dx12::make_dx12_context(&_adapter, &device, &queue) },
            #[cfg(all(target_family = "unix", not(target_vendor = "apple")))]
            Self::Vulkan => unsafe { vulkan::make_vulkan_context(&device, &queue) },
        }
    }

    fn make_surface(
        &self,
        size: PhysicalWindowSize,
        gr_context: &mut skia_safe::gpu::DirectContext,
        frame: &wgpu::SurfaceTexture,
    ) -> Option<skia_safe::Surface> {
        match self {
            #[cfg(target_vendor = "apple")]
            Self::Metal => unsafe { metal::make_metal_surface(size, gr_context, frame) },
            #[cfg(target_family = "windows")]
            Self::Dx12 => unsafe { dx12::make_dx12_surface(size, gr_context, frame) },
            #[cfg(all(target_family = "unix", not(target_vendor = "apple")))]
            Self::Vulkan => unsafe { vulkan::make_vulkan_surface(size, gr_context, frame) },
        }
    }

    fn import_texture(
        &self,
        canvas: &skia_safe::Canvas,
        texture: wgpu::Texture,
    ) -> Option<skia_safe::Image> {
        match self {
            #[cfg(target_vendor = "apple")]
            Self::Metal => unsafe { metal::import_metal_texture(canvas, texture) },
            #[cfg(target_family = "windows")]
            Self::Dx12 => unsafe { dx12::import_dx12_texture(canvas, texture) },
            #[cfg(all(target_family = "unix", not(target_vendor = "apple")))]
            Self::Vulkan => unsafe { vulkan::import_vulkan_texture(canvas, texture) },
        }
    }
}
