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
    backend: Backend,
    /// gsplit: GSPLIT_UI_PERF=1 aggregate instrumentation (see ui-render-
    /// decoupling-strategy.md Stage 0).
    ui_perf: RefCell<Option<UiPerf>>,
    /// gsplit: frames rendered since the last surface (re)configure. Freshly
    /// configured swapchains contain garbage; partial rendering must see
    /// buffer age 0 (= full repaint) until the whole buffer chain was painted.
    frames_since_configure: std::cell::Cell<u32>,
}

/// gsplit Stage-0 measurement: per-second aggregates of Skia draw time and
/// damage area, logged when GSPLIT_UI_PERF=1. The damage percentage is
/// self-calibrating: it's relative to the largest dirty area ever observed
/// (the first frame is always a full repaint), so no scale-factor plumbing
/// is needed.
struct UiPerf {
    last_log: std::time::Instant,
    frames: u32,
    full_repaints: u32,
    draw_us_sum: u64,
    draw_us_max: u64,
    dirty_area_sum: f64,
    max_area: f64,
}

impl UiPerf {
    fn new_if_enabled() -> Option<Self> {
        std::env::var("GSPLIT_UI_PERF").map_or(false, |v| v == "1").then(|| UiPerf {
            last_log: std::time::Instant::now(),
            frames: 0,
            full_repaints: 0,
            draw_us_sum: 0,
            draw_us_max: 0,
            dirty_area_sum: 0.0,
            max_area: 0.0,
        })
    }

    fn record(&mut self, draw_us: u64, dirty: &Option<DirtyRegion>) {
        self.frames += 1;
        self.draw_us_sum += draw_us;
        self.draw_us_max = self.draw_us_max.max(draw_us);
        match dirty {
            Some(region) => {
                let area: f64 = region.iter().map(|b| b.area() as f64).sum();
                self.max_area = self.max_area.max(area);
                self.dirty_area_sum += area;
            }
            None => {
                // Renderer reported no region = full repaint.
                self.full_repaints += 1;
                self.dirty_area_sum += self.max_area;
            }
        }
        if self.last_log.elapsed().as_secs() >= 1 {
            let f = self.frames.max(1) as f64;
            eprintln!(
                "gsplit ui-perf: {} fps · skia draw avg {:.2} ms max {:.2} ms · dirty avg {:.0}% of window · {} full repaints",
                self.frames,
                self.draw_us_sum as f64 / f / 1000.0,
                self.draw_us_max as f64 / 1000.0,
                100.0 * self.dirty_area_sum / f / self.max_area.max(1.0),
                self.full_repaints,
            );
            let max_area = self.max_area;
            *self = UiPerf {
                last_log: std::time::Instant::now(),
                frames: 0,
                full_repaints: 0,
                draw_us_sum: 0,
                draw_us_max: 0,
                dirty_area_sum: 0.0,
                max_area,
            };
        }
    }
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
            backend,
            ui_perf: RefCell::new(UiPerf::new_if_enabled()),
            frames_since_configure: std::cell::Cell::new(0),
        })
    }

    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn resize_event(&self, size: PhysicalWindowSize) -> Result<(), PlatformError> {
        {
            // gsplit: skip same-size reconfigures. Wayland delivers a final
            // configure on resize-mouseup at the unchanged size; reconfiguring
            // would replace the swapchain with garbage buffers while the UI has
            // no damage — under partial rendering only the few dirty items got
            // painted onto the garbage (the magenta-window-on-mouseup bug).
            let surface_config = self.surface_config.borrow();
            if surface_config.width == size.width && surface_config.height == size.height {
                return Ok(());
            }
        }
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
        // New swapchain = garbage buffers: report buffer age 0 (full repaint)
        // until every buffer in the assumed-triple-buffered chain was painted.
        self.frames_since_configure.set(0);
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
                // gsplit: fresh swapchain — see frames_since_configure.
                self.frames_since_configure.set(0);
                self.surface.get_current_texture().map_err(|e| {
                    format!("Error obtaining current surface texture after initial error: {e}")
                })?
            }
        };
        let skia_surface = self.backend.make_surface(size, gr_context, &frame);

        let mut skia_surface = skia_surface
            .ok_or_else(|| PlatformError::from("Failed to create Skia surface from WGPU"))?;

        // wgpu doesn't expose EGL_EXT_buffer_age, so assume triple buffering
        // (worst-case for FIFO/AutoVsync). This enables partial rendering to
        // union the last 2 frames' dirty regions instead of repainting everything.
        // gsplit: EXCEPT right after a (re)configure — new swapchain buffers
        // hold garbage, so report age 0 (full repaint) until the whole chain
        // was painted once (the magenta-on-resize-mouseup bug).
        let fsc = self.frames_since_configure.get();
        let age = if fsc >= 3 { 3 } else { 0 };
        self.frames_since_configure.set(fsc.saturating_add(1));
        let draw_start = std::time::Instant::now();
        let dirty = callback(skia_surface.canvas(), Some(gr_context), age);
        if let Some(perf) = self.ui_perf.borrow_mut().as_mut() {
            perf.record(draw_start.elapsed().as_micros() as u64, &dirty);
        }

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
