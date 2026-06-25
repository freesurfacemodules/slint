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
    /// gsplit Stage 2/3 (ui-render-decoupling): when set, this window is in
    /// INVERTED mode — Slint's UI renders into a persistent wgpu-texture-backed
    /// overlay (premultiplied alpha, partial-rendered by damage), and the hook
    /// composes [content under, UI overlay on top] into the swapchain instead
    /// of Skia. Inverted windows should use a transparent root background so
    /// the overlay only covers actual UI.
    compose_hook: RefCell<Option<ComposeHook>>,
    /// The UI overlay: a wgpu texture wrapped as a Skia render target.
    inverted_ui: RefCell<Option<InvertedUi>>,
}

/// gsplit: persistent UI-overlay target for an inverted window. The TEXTURE
/// persists (content accumulates across frames); the Skia surface wrapping it
/// is created FRESH each frame, like the swapchain path does — Slint's
/// partial renderer applies its dirty-region clip to the canvas, and clips
/// intersect cumulatively on a reused canvas (two disjoint frames' regions →
/// empty clip → every draw silently discarded; symptom: UI frozen after the
/// first paint).
struct InvertedUi {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
    /// A fresh overlay holds nothing: the first frame must report buffer
    /// age 0 (full repaint). The window rendered through the normal path
    /// before the hook registered, so Slint's items are already clean —
    /// at age 1 only incremental damage would ever land in the overlay
    /// (symptom: static UI missing, only per-frame elements visible).
    fresh: bool,
}

/// gsplit: everything the app's compose pass needs to draw one frame of an
/// inverted window into the swapchain. The hook encodes + submits its own
/// command buffer(s); the surface presents afterwards.
pub struct ComposeCtx<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    /// Swapchain texture view to render into.
    pub target: &'a wgpu::TextureView,
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
    /// The window's UI overlay (premultiplied alpha; same size as the target),
    /// to be composited on top of the hook's own content. None until the
    /// overlay exists (created on the first inverted frame).
    pub ui: Option<&'a wgpu::TextureView>,
}

/// gsplit: per-window compose callback (see [`ComposeCtx`]).
pub type ComposeHook = Box<dyn FnMut(&ComposeCtx<'_>)>;

impl WGPUSurface {
    /// gsplit Stage 2: install/remove the compose hook (inverted mode).
    pub fn set_compose_hook(&self, hook: Option<ComposeHook>) {
        if hook.is_none() {
            self.inverted_ui.borrow_mut().take();
        }
        *self.compose_hook.borrow_mut() = hook;
    }
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

    fn record(&mut self, draw_us: u64, dirty: &Option<DirtyRegion>, size: (u32, u32)) {
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
                "gsplit ui-perf [{}x{}]: {} fps · skia draw avg {:.2} ms max {:.2} ms · dirty avg {:.0}% of window · {} full repaints",
                size.0,
                size.1,
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
        // gsplit: configure the INITIAL swapchain identically to resize_event's
        // reconfigure. The default config produced a different buffer chain than
        // the post-resize one — under partial rendering the startup chain showed
        // stale-frame flicker on hover-after-UI-change, which permanently
        // disappeared after the first resize (= first AutoVsync reconfigure).
        surface_config.present_mode = wgpu::PresentMode::AutoVsync;
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
            compose_hook: RefCell::new(None),
            inverted_ui: RefCell::new(None),
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

        // gsplit Stage 2/3: inverted mode — the app's compose hook draws this
        // window's content into the swapchain; Slint's UI renders into a
        // persistent wgpu-texture-backed overlay (premultiplied alpha,
        // partial-rendered by damage) that the hook composites on top. No
        // Skia work touches the swapchain at all.
        if self.compose_hook.borrow().is_some() {
            {
                let mut ui = self.inverted_ui.borrow_mut();
                let needs_new =
                    ui.as_ref().map_or(true, |u| u.size != (size.width, size.height));
                if needs_new {
                    let format = self.surface_config.borrow().format;
                    let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("gsplit inverted UI overlay"),
                        size: wgpu::Extent3d {
                            width: size.width.max(1),
                            height: size.height.max(1),
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    });
                    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                    // Initialize through wgpu with a real clear pass. Two birds:
                    // (1) wgpu lazily ZERO-INITIALIZES textures on first tracked
                    // use — Skia's raw-Vulkan writes are invisible to wgpu, so
                    // without this the first compose sample would zero-clear the
                    // overlay, destroying everything Skia painted before it
                    // (symptom: static UI missing, only per-frame-damaged
                    // elements visible); (2) the pass leaves the texture in
                    // color-target state, which Skia's wrap declares.
                    let mut encoder = self.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor { label: Some("UI overlay init") },
                    );
                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("UI overlay init clear"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        ..Default::default()
                    });
                    encoder.transition_resources(
                        std::iter::empty(),
                        std::iter::once(wgpu::TextureTransition {
                            texture: &texture,
                            selector: None,
                            state: wgpu::TextureUses::COLOR_TARGET,
                        }),
                    );
                    self.queue.submit(Some(encoder.finish()));
                    *ui = Some(InvertedUi {
                        texture,
                        view,
                        size: (size.width, size.height),
                        fresh: true,
                    });
                }
                let ui = ui.as_mut().unwrap();
                // wgpu last tracked the overlay as RESOURCE (sampled by the
                // previous compose); move it back to color-attachment for Skia.
                let mut encoder =
                    self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("UI overlay to color target"),
                    });
                encoder.transition_resources(
                    std::iter::empty(),
                    std::iter::once(wgpu::TextureTransition {
                        texture: &ui.texture,
                        selector: None,
                        state: wgpu::TextureUses::COLOR_TARGET,
                    }),
                );
                self.queue.submit(Some(encoder.finish()));
                // Fresh Skia wrap each frame (see InvertedUi doc); the texture
                // content persists, so buffer age 1 (only new damage repaints),
                // except the first frame after creation (see fresh).
                let mut skia = self
                    .backend
                    .make_surface(size, gr_context, &ui.texture)
                    .ok_or_else(|| {
                        PlatformError::from("Failed to wrap UI overlay texture for Skia")
                    })?;
                let age = if ui.fresh { 0 } else { 1 };
                ui.fresh = false;
                // gsplit perf: record the overlay paint (draw time + dirty
                // region) so the inverted path reports the same ui-perf line as
                // the non-inverted one. Answers "does anything re-dirty the UI
                // every frame once a layer exists?" — dirty avg ~0% + 0 full
                // repaints means Slint elides the paint and the per-frame cost
                // is just present machinery.
                let draw_start = std::time::Instant::now();
                let dirty = callback(skia.canvas(), Some(gr_context), age);
                if let Some(perf) = self.ui_perf.borrow_mut().as_mut() {
                    perf.record(draw_start.elapsed().as_micros() as u64, &dirty, (size.width, size.height));
                }
            }
            // Transition any textures Slint sampled (imported images in the UI).
            let textures_to_transition = self.textures_to_transition_for_sampling.take();
            if !textures_to_transition.is_empty() {
                let mut encoder =
                    self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

            // Debug tool (kept from Stage-4 bring-up — it found the opaque
            // PreviewPane root): GSPLIT_OVERLAY_DEBUG=1 dumps the overlay once
            // to /tmp/overlay.raw (BGRA, rows padded to 256B). Inspect with:
            // ffmpeg -f rawvideo -pixel_format bgra -video_size <bpr/4>x<h>
            //   -i /tmp/overlay.raw -vf crop=<w>:<h>:0:0[,alphaextract] out.png
            if std::env::var("GSPLIT_OVERLAY_DEBUG").map_or(false, |v| v == "1") {
                use std::sync::atomic::{AtomicU32, Ordering};
                static DBG: AtomicU32 = AtomicU32::new(0);
                if DBG.fetch_add(1, Ordering::Relaxed) == 30 {
                    if let Some(u) = self.inverted_ui.borrow().as_ref() {
                        let bpr = (u.size.0 * 4).div_ceil(256) * 256;
                        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("overlay dump"),
                            size: (bpr * u.size.1) as u64,
                            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                            mapped_at_creation: false,
                        });
                        let mut enc = self.device.create_command_encoder(&Default::default());
                        enc.copy_texture_to_buffer(
                            wgpu::TexelCopyTextureInfo {
                                texture: &u.texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::TexelCopyBufferInfo {
                                buffer: &buf,
                                layout: wgpu::TexelCopyBufferLayout {
                                    offset: 0,
                                    bytes_per_row: Some(bpr),
                                    rows_per_image: None,
                                },
                            },
                            wgpu::Extent3d {
                                width: u.size.0,
                                height: u.size.1,
                                depth_or_array_layers: 1,
                            },
                        );
                        self.queue.submit(Some(enc.finish()));
                        let slice = buf.slice(..);
                        slice.map_async(wgpu::MapMode::Read, |_| {});
                        let _ = self.device.poll(wgpu::PollType::Wait {
                            submission_index: None,
                            timeout: None,
                        });
                        let data = slice.get_mapped_range();
                        std::fs::write("/tmp/overlay.raw", &*data).ok();
                        eprintln!(
                            "overlay-debug: dumped {}x{} bpr {} to /tmp/overlay.raw",
                            u.size.0, u.size.1, bpr
                        );
                    }
                }
            }

            let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
            {
                let cfg = self.surface_config.borrow();
                let ui = self.inverted_ui.borrow();
                let ctx = ComposeCtx {
                    device: &self.device,
                    queue: &self.queue,
                    target: &view,
                    format: cfg.format,
                    width: cfg.width,
                    height: cfg.height,
                    // The compose pass samples the overlay through normal wgpu
                    // usage tracking, which inserts the render→sample barrier.
                    ui: ui.as_ref().map(|u| &u.view),
                };
                (self.compose_hook.borrow_mut().as_mut().unwrap())(&ctx);
            }

            if let Some(pre_present_callback) = pre_present_callback.borrow_mut().as_mut() {
                pre_present_callback();
            }
            frame.present();
            return Ok(());
        }

        let skia_surface = self.backend.make_surface(size, gr_context, &frame.texture);

        let mut skia_surface = skia_surface
            .ok_or_else(|| PlatformError::from("Failed to create Skia surface from WGPU"))?;

        // gsplit: this non-inverted path runs only for windows WITHOUT a
        // compose hook — i.e. ordinary Slint windows like the Options dialog
        // (inverted windows take the compose-hook branch above). Those windows
        // have no live-texture content, so partial rendering buys nothing here
        // — and across platforms its buffer-age heuristics left unpainted
        // back-buffer regions showing garbage (intermittent magenta on open /
        // tab switch, worst on macOS where the swapchain buffer count differs
        // from the Wayland assumption). Always report buffer age 0 (full
        // repaint) so every frame paints the whole swapchain. The perf-critical
        // windows are all inverted and unaffected.
        let _ = self.frames_since_configure.get();
        let age = 0;
        let draw_start = std::time::Instant::now();
        let dirty = callback(skia_surface.canvas(), Some(gr_context), age);
        if let Some(perf) = self.ui_perf.borrow_mut().as_mut() {
            perf.record(draw_start.elapsed().as_micros() as u64, &dirty, (size.width, size.height));
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

    // gsplit Stage 3: takes any wgpu texture (not just swapchain frames) so the
    // inverted path can wrap its UI-overlay texture as a Skia render target.
    fn make_surface(
        &self,
        size: PhysicalWindowSize,
        gr_context: &mut skia_safe::gpu::DirectContext,
        texture: &wgpu::Texture,
    ) -> Option<skia_safe::Surface> {
        match self {
            #[cfg(target_vendor = "apple")]
            Self::Metal => unsafe { metal::make_metal_surface(size, gr_context, texture) },
            #[cfg(target_family = "windows")]
            Self::Dx12 => unsafe { dx12::make_dx12_surface(size, gr_context, texture) },
            #[cfg(all(target_family = "unix", not(target_vendor = "apple")))]
            Self::Vulkan => unsafe { vulkan::make_vulkan_surface(size, gr_context, texture) },
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
