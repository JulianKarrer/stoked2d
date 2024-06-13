use speedy2d::window::{WindowCreationOptions, WindowPosition};
use speedy2d::Window;
use std::sync::atomic::Ordering::Relaxed;
use std::thread;
use stoked2d::gui::gui::StokedWindowHandler;
use stoked2d::*;

fn main() {
    let window = Window::new_with_options(
        "Stoked 2D",
        WindowCreationOptions::new_windowed(
            speedy2d::window::WindowSize::PhysicalPixels(
                (WINDOW_SIZE[0].load(Relaxed), WINDOW_SIZE[1].load(Relaxed)).into(),
            ),
            Some(WindowPosition::Center),
        )
        .with_maximized(true),
    )
    .unwrap();

    thread::spawn(|| {
        if cfg!(feature = "gpu") {
            while gpu_version::gpu::run(None) {}
        } else {
            while simulation::run(None, "scenes/setting-pdrop-laminar.png") {}
            // while simulation::run(None, "scenes/setting-dambreak.png") {}
            // while simulation::run(None, "scenes/setting_column.png") {}
            // while simulation::run(None, "scenes/setting_column_small.png") {}
        }
    });
    window.run_loop(egui_speedy2d::WindowWrapper::new(StokedWindowHandler {}));

    // while gpu_version::gpu::run(Some(10.0)) {}
}
