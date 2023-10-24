
use crate::*;

use egui_speedy2d::egui;
use speedy2d::shape::Rectangle;
use speedy2d::window::MouseScrollDistance;
use speedy2d::{window::{WindowHelper, self}, Graphics2D, color::Color, dimen::Vector2};
use atomic::Atomic;

// GUI RELATED CONSTANTS AND ATOMICS
static ZOOM:AtomicF32 = AtomicF32::new(20.0);
const ZOOM_SPEED:f32 = 1.5;
static DRAGGING:AtomicBool = AtomicBool::new(false);
const BOUNDARY_THCKNESS:f64 = 0.05;
static GUI_FPS:AtomicF64 = AtomicF64::new(60.0);

lazy_static! {
  static ref DRAG_OFFSET:Arc<RwLock<speedy2d::dimen::Vec2>> = Arc::new(RwLock::new(speedy2d::dimen::Vec2::new(0.0, 0.0)));
  static ref DRAG_LAST:Arc<RwLock<Option<speedy2d::dimen::Vec2>>> = Arc::new(RwLock::new(None));
  static ref LAST_FRAME_TIME:Atomic<u128> = Atomic::new(0);
}

fn camera_transform(p: &DVec2, offset: &Vector2<f32>, zoom: f32, width: f32, height: f32)->Vector2<f32>{
  Vector2::new(
    zoom*p.x as f32 + offset.x + width*0.5  , 
   -zoom*p.y as f32 + offset.y + height*0.5 )
}

pub struct StokedWindowHandler;
impl egui_speedy2d::WindowHandler for StokedWindowHandler {
  // MAIN RENDERING LOOP
  fn on_draw(
    &mut self,
    helper: &mut WindowHelper,
    graphics: &mut Graphics2D,
    egui_ctx: &egui::Context,
  ) {
    // update FPS counter in title
    let now = timestamp();
    let dt = micros_to_seconds(now-LAST_FRAME_TIME.load(Relaxed));
    GUI_FPS.store(GUI_FPS.load(Relaxed)*FPS_SMOOTING + 1.0/dt * (1.0-FPS_SMOOTING), Relaxed);
    LAST_FRAME_TIME.store(now, Relaxed);
    helper.set_title(format!("Stoked 2D  -  GUI: {:.1} FPS  -  SIM: {:.1} FPS", GUI_FPS.load(Relaxed), SIM_FPS.load(Relaxed)));

    // clear screen
    graphics.clear_screen(Color::BLACK);

    let (w,h) = (WINDOW_SIZE[0].load(Relaxed) as f32, WINDOW_SIZE[1].load(Relaxed) as f32);
    let z = ZOOM.load(Relaxed);
    let off = *(*DRAG_OFFSET).read();

    // draw the boundary
    graphics.draw_rectangle(Rectangle::new(
      camera_transform(&BOUNDARY[0], &off, z, w, h), 
      camera_transform(&BOUNDARY[1], &off, z, w, h), 
      ), 
      Color::WHITE
    );
    graphics.draw_rectangle(Rectangle::new(
      camera_transform(&(BOUNDARY[0]+DVec2::ONE*BOUNDARY_THCKNESS), &off, z, w, h), 
      camera_transform(&(BOUNDARY[1]-DVec2::ONE*BOUNDARY_THCKNESS), &off, z, w, h), 
      ), 
      Color::BLACK
    );

    // draw each particle, with (0,0) being the centre of the screen
    let gradient = colorgrad::spectral();
    (*HISTORY).read().last().unwrap().0.iter().zip((*COLOUR).read().iter()).for_each(|(p, c)|{
      let colour = gradient.at(*c);
      graphics.draw_circle(
        camera_transform(p, &off, z, w, h), 
        0.5*z*H as f32, 
        Color::from_rgb(colour.r as f32, colour.g as f32, colour.b as f32)
      )
    });
    
    // draw the GUI
    let mut gravity = GRAVITY.load(Relaxed);
    let mut restart = false;
    egui::Window::new("Settings").resizable(true).show(egui_ctx, |ui| {
      // adjust gravity
      if ui.button("Restart Simulation").clicked(){
        restart = true;
      }
      ui.horizontal(|ui| {
        ui.label("Gravity");
        ui.add(egui::DragValue::new(&mut gravity).speed(0.1).max_decimals(3));
      });
      
    });
    GRAVITY.store(gravity, Relaxed);
    REQUEST_RESTART.store(restart, Relaxed);
    helper.request_redraw();
  }

  fn on_resize(
      &mut self,
      _helper: &mut WindowHelper<()>,
      size_pixels: speedy2d::dimen::UVec2,
      _egui_ctx: &egui::Context,
    ) {
    WINDOW_SIZE[0].store(size_pixels.x, Relaxed);
    WINDOW_SIZE[1].store(size_pixels.y, Relaxed);
  }

  // zooming
  fn on_mouse_wheel_scroll(
      &mut self,
      _helper: &mut WindowHelper<()>,
      distance: window::MouseScrollDistance,
      _egui_ctx: &egui::Context,
    ) {
      let mut z = ZOOM.load(Relaxed);
      z = (z + match distance{
        MouseScrollDistance::Lines { x: _,y, z: _ } => y as f32 * ZOOM_SPEED,
        _ => 0.0,
      }).max(1.0);
    ZOOM.store(z,Relaxed);
  }

  // dragging the window
  fn on_mouse_button_down(
      &mut self,
      _helper: &mut WindowHelper<()>,
      button: window::MouseButton,
      _egui_ctx: &egui::Context,
    ) {
      if button == window::MouseButton::Right {
        DRAGGING.store(true, Relaxed)
      }
  }

  fn on_mouse_button_up(
      &mut self,
      _helper: &mut WindowHelper<()>,
      button: window::MouseButton,
      _egui_ctx: &egui::Context,
    ) {
    if button == window::MouseButton::Right {
      DRAGGING.store(false, Relaxed);
      *(*DRAG_LAST).write() = None;
    }
  }

  fn on_mouse_move(
      &mut self,
      _helper: &mut WindowHelper<()>,
      position: speedy2d::dimen::Vec2,
      _egui_ctx: &egui::Context,
    ) 
  {
    if DRAGGING.load(Relaxed){
      let last = *(*DRAG_LAST).read();
      if last.is_none() {
        *(*DRAG_LAST).write() = Some(position);
      } 
      *(*DRAG_OFFSET).write() += position - (*DRAG_LAST).read().unwrap();
      *(*DRAG_LAST).write() = Some(position);
    }
  }
}