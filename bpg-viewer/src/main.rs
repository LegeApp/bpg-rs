// Minimal, fast BPG image viewer.
//
// Goals: instant startup, load a file passed on the command line (or dropped in)
// as fast as possible, and put essentially the whole window to the image. Any
// decode failure is reported clearly in-window instead of crashing.

mod decoder;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use eframe::egui;
use egui::{Color32, ColorImage, Pos2, Rect, Sense, TextureHandle, TextureOptions, Vec2};

fn main() -> eframe::Result<()> {
    // First non-flag argument is the file to open.
    let initial_path = std::env::args().skip(1).find(|a| !a.starts_with('-')).map(PathBuf::from);

    let title = match &initial_path {
        Some(p) => format!("{} — BPG Viewer", file_label(p)),
        None => "BPG Viewer".to_string(),
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 700.0])
            .with_min_inner_size([240.0, 160.0])
            .with_title(title)
            .with_app_id("bpg-view"),
        ..Default::default()
    };

    eframe::run_native(
        "bpg-view",
        options,
        Box::new(|_cc| Ok(Box::new(ViewerApp::new(initial_path)))),
    )
}

/// Pixel data decoded from a file, ready to be uploaded as a texture.
struct PendingImage {
    rgba: Vec<u8>,
    width: usize,
    height: usize,
    path: PathBuf,
}

/// An image currently displayed, with its view transform.
struct ImageView {
    texture: TextureHandle,
    size: Vec2,
    zoom: f32,
    pan: Vec2,
    fitted: bool,
}

enum Content {
    Empty,
    Error { path: Option<PathBuf>, message: String },
    Image(ImageView),
}

struct ViewerApp {
    content: Content,
    pending: Option<PendingImage>,
    title_dirty: bool,
    // egui builds its font atlas one frame after text is first requested, and a
    // static error/empty screen otherwise requests no repaint — so the message
    // would never paint. Force a few extra frames after every state change.
    frames_to_paint: u32,
}

impl ViewerApp {
    fn new(initial: Option<PathBuf>) -> Self {
        let mut app =
            ViewerApp { content: Content::Empty, pending: None, title_dirty: false, frames_to_paint: 3 };
        if let Some(path) = initial {
            app.request_load(path);
        }
        app
    }

    /// Decode a file off the GUI's critical path. Never panics out: a decoder
    /// panic is caught and surfaced as an error, so a bad file can't take the
    /// window down.
    fn request_load(&mut self, path: PathBuf) {
        self.title_dirty = true;
        self.frames_to_paint = 3;
        let result = catch_unwind(AssertUnwindSafe(|| decode_to_rgba(&path)));
        match result {
            Ok(Ok(pending)) => {
                self.pending = Some(pending);
            }
            Ok(Err(message)) => {
                eprintln!("bpg-view: {}: {}", path.display(), message);
                self.content = Content::Error { path: Some(path), message };
            }
            Err(_) => {
                let message = "internal decoder error (panic) while decoding".to_string();
                eprintln!("bpg-view: {}: {}", path.display(), message);
                self.content = Content::Error { path: Some(path), message };
            }
        }
    }
}

/// Decode `path` to RGBA8. Returns a human-readable error string on failure.
fn decode_to_rgba(path: &Path) -> Result<PendingImage, String> {
    let path_str = path.to_str().ok_or_else(|| "path is not valid UTF-8".to_string())?;

    if !path.exists() {
        return Err("file does not exist".to_string());
    }

    let decoded = decoder::decode_file(path_str).map_err(|e| format!("cannot decode BPG: {e}"))?;

    let width = decoded.width as usize;
    let height = decoded.height as usize;
    // `decoded.data` is already RGBA8 — move it (no clone; matters at ~170 MP).
    let rgba = decoded.data;
    if rgba.len() < width * height * 4 {
        return Err("decoder returned truncated pixel data".to_string());
    }

    Ok(PendingImage { rgba, width, height, path: path.to_path_buf() })
}

fn file_label(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| path.display().to_string())
}

/// Downscale an RGBA8 image by the smallest integer factor that brings both
/// sides within `max_side` (the GPU's max texture dimension), box-averaging
/// each `f×f` block. Returns the original buffer unchanged when it already
/// fits. Aspect ratio is preserved (same factor on both axes).
fn fit_to_texture(rgba: &[u8], w: usize, h: usize, max_side: usize) -> (Vec<u8>, usize, usize) {
    if w <= max_side && h <= max_side {
        return (rgba.to_vec(), w, h);
    }
    let f = w.max(h).div_ceil(max_side).max(2);
    let (nw, nh) = (w / f, h / f);
    let mut out = vec![0u8; nw * nh * 4];
    let area = (f * f) as u32;
    for ny in 0..nh {
        for nx in 0..nw {
            let mut acc = [0u32; 4];
            for dy in 0..f {
                let row = (ny * f + dy) * w;
                for dx in 0..f {
                    let si = (row + nx * f + dx) * 4;
                    acc[0] += rgba[si] as u32;
                    acc[1] += rgba[si + 1] as u32;
                    acc[2] += rgba[si + 2] as u32;
                    acc[3] += rgba[si + 3] as u32;
                }
            }
            let di = (ny * nw + nx) * 4;
            out[di] = (acc[0] / area) as u8;
            out[di + 1] = (acc[1] / area) as u8;
            out[di + 2] = (acc[2] / area) as u8;
            out[di + 3] = (acc[3] / area) as u8;
        }
    }
    (out, nw, nh)
}

impl eframe::App for ViewerApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.08, 0.08, 0.08, 1.0]
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Upload any freshly decoded image as a texture (needs the egui context).
        if let Some(p) = self.pending.take() {
            // The GPU caps a single texture's side (commonly 16384). bpg-rs
            // decodes much larger stills (e.g. 18432-wide heightmaps), so
            // downscale to fit before upload — otherwise egui_glow panics.
            let max_side = ctx.input(|i| i.max_texture_side).max(2048);
            let (rgba, tw, th) = fit_to_texture(&p.rgba, p.width, p.height, max_side);
            let color = ColorImage::from_rgba_unmultiplied([tw, th], &rgba);
            let texture = ctx.load_texture("bpg-image", color, TextureOptions::LINEAR);
            let scaled = tw != p.width || th != p.height;
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!(
                "{} ({}×{}{}) — BPG Viewer",
                file_label(&p.path),
                p.width,
                p.height,
                if scaled { ", fit to GPU" } else { "" }
            )));
            self.content = Content::Image(ImageView {
                texture,
                size: Vec2::new(tw as f32, th as f32),
                zoom: 1.0,
                pan: Vec2::ZERO,
                fitted: false,
            });
        } else if self.title_dirty {
            if let Content::Error { path, .. } = &self.content {
                let name = path.as_deref().map(file_label).unwrap_or_default();
                ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!(
                    "{name} — BPG Viewer (decode failed)"
                )));
            }
        }
        self.title_dirty = false;

        // Global keys.
        if ctx.input(|i| i.key_pressed(egui::Key::Q) || i.key_pressed(egui::Key::Escape)) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }

        // Accept dropped files (take the first BPG-looking one).
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if let Some(file) = dropped.into_iter().find_map(|f| f.path) {
            self.request_load(file);
        }

        if self.frames_to_paint > 0 {
            self.frames_to_paint -= 1;
            ctx.request_repaint();
        }

        let frame = egui::Frame::default().fill(Color32::from_rgb(20, 20, 20));
        egui::CentralPanel::default().frame(frame).show(ctx, |ui| match &mut self.content {
            Content::Image(view) => draw_image(ui, view),
            Content::Error { path, message } => draw_error(ui, path.as_deref(), message),
            Content::Empty => draw_empty(ui),
        });
    }
}

fn draw_image(ui: &mut egui::Ui, view: &mut ImageView) {
    let rect = ui.available_rect_before_wrap();

    // Fit to window on first display (and clamp absurd zooms).
    if !view.fitted && view.size.x > 0.0 && view.size.y > 0.0 {
        let scale = (rect.width() / view.size.x).min(rect.height() / view.size.y);
        view.zoom = scale.clamp(0.01, 32.0);
        view.pan = Vec2::ZERO;
        view.fitted = true;
    }

    let response = ui.allocate_rect(rect, Sense::click_and_drag());

    // Drag to pan.
    if response.dragged() {
        view.pan += response.drag_delta();
    }

    // Scroll / +/- to zoom (around the window center).
    let mut zoom_factor = 1.0;
    let scroll = ui.input(|i| i.raw_scroll_delta.y);
    if scroll != 0.0 {
        zoom_factor *= 1.0 + scroll * 0.0015;
    }
    if ui.input(|i| i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals)) {
        zoom_factor *= 1.2;
    }
    if ui.input(|i| i.key_pressed(egui::Key::Minus)) {
        zoom_factor /= 1.2;
    }
    if zoom_factor != 1.0 {
        view.zoom = (view.zoom * zoom_factor).clamp(0.01, 32.0);
    }

    // F = fit, 1 = actual size.
    if ui.input(|i| i.key_pressed(egui::Key::F)) {
        view.fitted = false;
    }
    if ui.input(|i| i.key_pressed(egui::Key::Num1)) {
        view.zoom = 1.0;
        view.pan = Vec2::ZERO;
    }

    let display = view.size * view.zoom;
    let center = rect.center() + view.pan;
    let image_rect = Rect::from_center_size(center, display);

    ui.painter().image(
        view.texture.id(),
        image_rect,
        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
        Color32::WHITE,
    );
}

fn draw_error(ui: &mut egui::Ui, path: Option<&Path>, message: &str) {
    let rect = ui.available_rect_before_wrap();
    let center = rect.center();
    let painter = ui.painter();

    painter.text(
        center - Vec2::new(0.0, 28.0),
        egui::Align2::CENTER_CENTER,
        "⚠  Could not open image",
        egui::FontId::proportional(22.0),
        Color32::from_rgb(255, 120, 120),
    );
    if let Some(p) = path {
        painter.text(
            center,
            egui::Align2::CENTER_CENTER,
            p.display().to_string(),
            egui::FontId::proportional(14.0),
            Color32::from_rgb(200, 200, 200),
        );
    }
    painter.text(
        center + Vec2::new(0.0, 26.0),
        egui::Align2::CENTER_CENTER,
        message,
        egui::FontId::proportional(15.0),
        Color32::from_rgb(230, 180, 180),
    );
    painter.text(
        center + Vec2::new(0.0, 56.0),
        egui::Align2::CENTER_CENTER,
        "Drop another file to try again  ·  Q to quit",
        egui::FontId::proportional(12.0),
        Color32::from_rgb(140, 140, 140),
    );
}

fn draw_empty(ui: &mut egui::Ui) {
    let center = ui.available_rect_before_wrap().center();
    ui.painter().text(
        center,
        egui::Align2::CENTER_CENTER,
        "Drop a .bpg file here\nor: bpg-view <file.bpg>",
        egui::FontId::proportional(20.0),
        Color32::from_rgb(150, 150, 150),
    );
}
