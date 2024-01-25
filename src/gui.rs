use std::sync::Mutex;

use crate::*;
use crate::datastructure::Grid;
use crate::simulation::{Attributes, PressureEquation, Solver, average_density};
use atomic_enum::atomic_enum;
use egui_speedy2d::egui::{self, RichText, ImageButton, Vec2, Ui, TextureId};  
use egui::FontId;
use egui::plot::{Line, Plot, PlotPoints};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use speedy2d::shape::Rectangle;
use speedy2d::window::MouseScrollDistance;
use speedy2d::{window::{WindowHelper, self}, Graphics2D, color::Color, dimen::Vector2};
use atomic::Atomic;

// GUI RELATED SETTINGS
const FONT_HEADING_SIZE:f32 = 25.0;
static ICON_SIZE:Vec2 = Vec2::new(24.0, 24.0);

// GUI RELATED CONSTANTS AND ATOMICS
static ZOOM:AtomicF32 = AtomicF32::new(20.0);
const ZOOM_SPEED:f32 = 1.5;
static DRAGGING:AtomicBool = AtomicBool::new(false);
const BOUNDARY_THCKNESS:f64 = 0.05;
static GUI_FPS:AtomicF64 = AtomicF64::new(60.0);
static VISUALIZED_FEATURE:AtomicVisualizedFeature = AtomicVisualizedFeature::new(VisualizedFeature::Density);
static COLOUR_SCHEME:AtomicColourScheme = AtomicColourScheme::new(ColourScheme::Spectral);

lazy_static! {
  static ref DRAG_OFFSET:Arc<RwLock<speedy2d::dimen::Vec2>> = Arc::new(RwLock::new(speedy2d::dimen::Vec2::new(0.0, 0.0)));
  static ref DRAG_LAST:Arc<RwLock<Option<speedy2d::dimen::Vec2>>> = Arc::new(RwLock::new(None));
  static ref LAST_FRAME_TIME:Atomic<u128> = Atomic::new(0);
  static ref PLAY_STATE:Arc<RwLock<PlaybackState>> = Arc::new(RwLock::new(PlaybackState::CaughtUp));
  // image data for png icons:
  static ref IMAGE_PLAY:Arc<Mutex<Option<egui::TextureHandle>>> = Arc::new(Mutex::new(None));
  static ref IMAGE_PAUSE:Arc<Mutex<Option<egui::TextureHandle>>> = Arc::new(Mutex::new(None));
  static ref IMAGE_FORWARD:Arc<Mutex<Option<egui::TextureHandle>>> = Arc::new(Mutex::new(None));
  static ref IMAGE_REPLAY:Arc<Mutex<Option<egui::TextureHandle>>> = Arc::new(Mutex::new(None));
}

/// Stores the playback state of the GUI
#[derive(PartialEq)]
enum PlaybackState{
  Playing(u128),
  Paused(f64, usize),
  CaughtUp
}

/// Possible features that are visualized with colour
#[derive(PartialEq)]
#[atomic_enum]
enum VisualizedFeature {
  Density,
  SpaceFillingCurve,
}

/// Colour schemes for visualization
#[derive(PartialEq)]
#[atomic_enum]
enum ColourScheme {
  Spectral,
  Rainbow,
  Virdis,
}


/// Represents the features of the simulation that are stored for visualization
#[derive(Default)]
pub struct HistoryTimestep{
  pub pos: Vec<DVec2>,
  pub current_t: f64,
  pub densities: Vec<f64>,
  pub grid_handle_index: Vec<usize>
}

pub struct History{
  pub steps: Vec<HistoryTimestep>,
  plot_density: Vec<[f64;2]>
}

impl Default for History{
  fn default() -> Self {
    Self { steps: vec![HistoryTimestep::default()], plot_density: vec![[0.0, M/(H*H)]] }
  }
}

impl History{
  pub fn reset_and_add(&mut self, state: &Attributes, grid: &Grid, current_t: f64){
    self.steps.clear();
    self.plot_density.clear();
    self.add(state, grid, current_t);
  }

  pub fn add_step(&mut self, state: &Attributes, grid: &Grid, current_t: f64){
    self.add(state, grid, current_t)
  }

  fn add(&mut self, state: &Attributes, grid: &Grid, current_t: f64){
    let densities = state.den.clone();
    let average_density = average_density(&densities);
    self.plot_density.push([current_t, average_density]);
    self.steps.push(HistoryTimestep{ 
      pos: state.pos.clone(), 
      current_t, 
      densities, 
      grid_handle_index: grid.handles.par_iter().map(|x| x.index ).collect(),
    })
  }
}

/// Performs a camera transformation from a given point in world-space to drawing space
fn camera_transform(p: &DVec2, offset: &Vector2<f32>, zoom: f32, width: f32, height: f32)->Vector2<f32>{
  Vector2::new(
    zoom*p.x as f32 + offset.x + width*0.5  , 
   -zoom*p.y as f32 + offset.y + height*0.5 )
}

pub struct StokedWindowHandler;
impl egui_speedy2d::WindowHandler for StokedWindowHandler {
  fn on_start(
    &mut self,
    helper: &mut WindowHelper<()>,
    _info: window::WindowStartupInfo,
    _egui_ctx: &egui::Context,
  ) {
    helper.set_icon_from_rgba_pixels(vec![0u8;256*4], Vector2::new(16, 16)).unwrap();
  }

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
    // draw all boundary particles
    let boundary_colour = Color::from_rgba(1.0, 1.0, 1.0, 0.5);
    BOUNDARY_PARTICLES.read().iter().for_each(|p|{
      graphics.draw_circle(
        camera_transform(p, &off, z, w, h), 
        0.5*z*H as f32, 
        boundary_colour
      )
    });

    // get the current playstate and decide what to display
    let mut current_playback_t = 0.0;
    let mut current_step = 0;
    {
      let mut playstate = PLAY_STATE.write();
      let hist = (*HISTORY).read();
      let history_timestep = match *playstate{
        PlaybackState::Playing(start_timestep) => {
          current_playback_t = micros_to_seconds(timestamp() - start_timestep);
          let res = hist.steps.iter().enumerate().find(|(_, hts)| 
            hts.current_t >= current_playback_t
          );
          if let Some((i, hts)) = res {
            current_step = i;
            hts
          } else {
            *playstate = PlaybackState::CaughtUp;
            hist.steps.last().unwrap()
          }
        },
        PlaybackState::Paused(_, current_step) => &hist.steps[current_step],
        PlaybackState::CaughtUp => hist.steps.last().unwrap(),
      };
      // draw each particle, with (0,0) being the centre of the screen
      let scheme = COLOUR_SCHEME.load(Relaxed);
      let gradient = match scheme{
        ColourScheme::Spectral => colorgrad::spectral(),
        ColourScheme::Rainbow => colorgrad::rainbow(),
        ColourScheme::Virdis => colorgrad::viridis(),
      };
      let gradient_flipper = if scheme==ColourScheme::Virdis {1.0} else {-1.0};
      history_timestep.pos.iter().enumerate().for_each(|(i, p)|{
        let c = match VISUALIZED_FEATURE.load(Relaxed){
          VisualizedFeature::Density => {
            let (min, max) = (0.8, 1.2);
            (history_timestep.densities[i].min(max)-min)/(max-min)
          },
          VisualizedFeature::SpaceFillingCurve => {
            history_timestep.grid_handle_index[i] as f64 / history_timestep.pos.len() as f64
          },
        };
        let colour = gradient.at((c*2.0-1.0)*gradient_flipper*0.5+0.5);
        graphics.draw_circle(
          camera_transform(p, &off, z, w, h), 
          0.5*z*H as f32, 
          Color::from_rgb(colour.r as f32, colour.g as f32, colour.b as f32)
        )
      });
    }
    
    // draw the GUI
    let mut restart = false;
    let mut gravity = GRAVITY.load(Relaxed);
    let mut k: f64 = K.load(Relaxed);
    let mut nu: f64 = NU.load(Relaxed);
    let mut rho_0:f64 = RHO_ZERO.load(Relaxed);
    let mut pressure_eq:PressureEquation = PRESSURE_EQ.load(Relaxed);
    let mut solver:Solver = SOLVER.load(Relaxed);
    let mut max_delta_rho:f64 = MAX_RHO_DEVIATION.load(Relaxed);
    let mut lambda:f64 = LAMBDA.load(Relaxed);
    let mut max_dt:f64 = MAX_DT.load(Relaxed);
    let mut init_dt:f64 = INITIAL_DT.load(Relaxed);
    let mut resort:u32 = RESORT_ATTRIBUTES_EVERY_N.load(Relaxed);
    let mut curve:GridCurve = GRID_CURVE.load(Relaxed);
    let mut feature:VisualizedFeature = VISUALIZED_FEATURE.load(Relaxed);
    let mut colours:ColourScheme = COLOUR_SCHEME.load(Relaxed);
    // create fonts
    let header = FontId::proportional(FONT_HEADING_SIZE);
    // SETTINGS WINDOW
    egui::Window::new("Settings").resizable(true).show(egui_ctx, |ui| {
      // adjust, gravity, stiffness etc. 
      ui.label(RichText::new("Simulation").font(header.clone()));
      ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut lambda).speed(0.001).max_decimals(3).clamp_range(0.001..=1.0));
        ui.label("Timestep λ");
      });
      ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut init_dt).speed(0.0001).max_decimals(4).clamp_range(0.0001..=1.0));
        ui.label("Initial Δt");
      });
      ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut max_dt).speed(0.001).max_decimals(3).clamp_range(0.001..=1.0));
        ui.label("Maximum Δt");
      });
      ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut gravity).speed(0.1).max_decimals(3));
        ui.label("Gravity g");
      });
      ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut k).speed(10).max_decimals(0).clamp_range(0.0..=f64::MAX));
        ui.label("Stiffness k");
      });
      ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut nu).speed(0.001).max_decimals(3));
        ui.label("Viscosity ν");
      });
      ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut rho_0).speed(0.01).clamp_range(0.01..=100_000.0));
        ui.label("Rest density ρ₀");
      });
      // adjust pressure solver settings
      ui.separator();
      ui.label(RichText::new("Pressure Solver").font(header.clone()));
      egui::ComboBox::from_label("Pressure Equation")
        .selected_text(format!("{:?}", pressure_eq))
        .show_ui(ui, |ui: &mut Ui| {
          ui.selectable_value(&mut pressure_eq, PressureEquation::Relative, "Relative");
          ui.selectable_value(&mut pressure_eq, PressureEquation::ClampedRelative, "Clamped Relative");
          ui.selectable_value(&mut pressure_eq, PressureEquation::Compressible, "Compressible");
          ui.selectable_value(&mut pressure_eq, PressureEquation::ClampedCompressible, "Clamped Compressible");
          ui.selectable_value(&mut pressure_eq, PressureEquation::Absolute, "Absolute");
        }
      );
      egui::ComboBox::from_label("Solver")
        .selected_text(format!("{:?}", solver))
        .show_ui(ui, |ui: &mut Ui| {
          ui.selectable_value(&mut solver, Solver::SESPH, "SESPH");
          ui.selectable_value(&mut solver, Solver::SSESPH, "SSESPH");
          ui.selectable_value(&mut solver, Solver::ISESPH, "ISESPH");
        }
      );
      ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut max_delta_rho).speed(0.01).max_decimals(3).clamp_range(0.01..=0.1));
        ui.label("Max. |Δρ| η");
      });
      // adjust datastructure settings
      ui.separator();
      ui.label(RichText::new("Datastructure").font(header.clone()));
      ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut resort).speed(1));
        ui.label("Resort every N");
      });
      egui::ComboBox::from_label("Space filling curve")
        .selected_text(format!("{:?}", curve))
        .show_ui(ui, |ui| {
          ui.selectable_value(&mut curve, GridCurve::Morton, "Morton");
          ui.selectable_value(&mut curve, GridCurve::Hilbert, "Hilbert");
        }
      );
      // adjust GUI
      ui.separator();
      ui.label(RichText::new("Visualization").font(header.clone()));
      egui::ComboBox::from_label("Visualized feature")
        .selected_text(format!("{:?}", feature))
        .show_ui(ui, |ui| {
          ui.selectable_value(&mut feature, VisualizedFeature::Density, "Density");
          ui.selectable_value(&mut feature, VisualizedFeature::SpaceFillingCurve, "Space Filling Curve");
        }
      );
      egui::ComboBox::from_label("Colour Scheme")
        .selected_text(format!("{:?}", colours))
        .show_ui(ui, |ui| {
          ui.selectable_value(&mut colours, ColourScheme::Spectral, "Spectral");
          ui.selectable_value(&mut colours, ColourScheme::Rainbow, "Rainbow");
          ui.selectable_value(&mut colours, ColourScheme::Virdis, "Virdis");
        }
      );

      // PLOT
      ui.separator();
      ui.label(RichText::new("Plot").font(header.clone()));
      let avg_den_plot: PlotPoints = { (*HISTORY).read().plot_density.clone().into()};
      Plot::new("plot").view_aspect(2.0).show(ui, |plot_ui| 
        plot_ui.line(Line::new(avg_den_plot))
      );
      
    });  
    // BOTTOM PANEL
    egui::panel::TopBottomPanel::bottom("time_panel").resizable(false).show(egui_ctx, |ui|{
      ui.horizontal(|ui| {
        // show  playback time control icons
        let (play, _) = get_image(IMAGE_PLAY.as_ref(), "./assets/play.png", "play", ui);
        let (pause, _) = get_image(IMAGE_PAUSE.as_ref(), "./assets/pause.png", "play", ui);
        let (forward, _) = get_image(IMAGE_FORWARD.as_ref(), "./assets/forward.png", "play", ui);
        let (replay, _) = get_image(IMAGE_REPLAY.as_ref(), "./assets/replay.png", "play", ui);
        let mut playstate = PLAY_STATE.write();
        if ui.add(ImageButton::new(replay, ICON_SIZE))
          .on_hover_text("Restart")
          .clicked() {
          *playstate = PlaybackState::CaughtUp;
          restart = true;
        }
        if ui.add(ImageButton::new(
          match *playstate {
            PlaybackState::Playing(_) => pause,
            PlaybackState::Paused(..) => play,
            PlaybackState::CaughtUp => play,
          }, ICON_SIZE))
          .on_hover_text("Play in real time")
          .clicked(){
          match *playstate{
            PlaybackState::Playing(_) => *playstate = PlaybackState::Paused(current_playback_t, current_step),
            PlaybackState::Paused(current_playback_t, _) => {
              *playstate = PlaybackState::Playing(timestamp() - seconds_to_micros(current_playback_t))
            },
            PlaybackState::CaughtUp => *playstate = PlaybackState::Playing(timestamp()),
          };
        }
        if ui.add(ImageButton::new(forward, ICON_SIZE))
          .on_hover_text("Skip to present")
          .clicked(){
          *playstate = PlaybackState::CaughtUp
        }
        // show available playback time
        let time_available:f64 = (*HISTORY).read().steps.last().unwrap().current_t;
        ui.add_sized([50.0,30.0], egui::Label::new(match *playstate {
          PlaybackState::Playing(_) => "playing",
          PlaybackState::Paused(_, _) => "paused",
          PlaybackState::CaughtUp => "caught up",
        }));
        let mut time_selected = match *playstate {
          PlaybackState::Playing(start) => micros_to_seconds(timestamp() - start),
          PlaybackState::Paused(current_t, _) => current_t,
          PlaybackState::CaughtUp => time_available,
        };
        ui.add_sized([50.0,30.0], egui::Label::new(format!("{:.2}/{:.2}", time_selected, time_available)));
        // adjust slider width
        let mut style: egui::Style = (*egui_ctx.style()).clone();
        style.spacing.slider_width = ui.available_width();
        egui_ctx.set_style(style);
        // show slider 
        if ui.add(egui::Slider::new(&mut time_selected, f64::EPSILON..=time_available).text("current t")).changed(){
          // scrubbing on the slider is only enabled when playback is paused
          if let PlaybackState::Paused(..) = *playstate{
            let hist = (*HISTORY).read();
            let res = hist.steps.iter().enumerate().find(|(_, hts)| 
              hts.current_t >= time_selected
            );
            if let Some((i, step)) = res {
              *playstate = PlaybackState::Paused(step.current_t, i)
            }
          }
        };
      });
    });

    // write back potentially modified atomics
    LAMBDA.store(lambda, Relaxed);
    MAX_DT.store(max_dt, Relaxed);
    INITIAL_DT.store(init_dt, Relaxed);
    GRAVITY.store(gravity, Relaxed);
    K.store(k, Relaxed);
    NU.store(nu, Relaxed);
    RHO_ZERO.store(rho_0, Relaxed);
    PRESSURE_EQ.store(pressure_eq, Relaxed);
    SOLVER.store(solver, Relaxed);
    MAX_RHO_DEVIATION.store(max_delta_rho, Relaxed);
    REQUEST_RESTART.store(restart, Relaxed);
    RESORT_ATTRIBUTES_EVERY_N.store(resort, Relaxed);
    GRID_CURVE.store(curve, Relaxed);
    VISUALIZED_FEATURE.store(feature, Relaxed);
    COLOUR_SCHEME.store(colours, Relaxed);

    // draw the new frame
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


// LOAD IMAGES AND MANAGE TEXTURES
fn get_image(image_data: &Mutex<Option<egui::TextureHandle>>, path: &str, name: &str, ui: &Ui)->(TextureId, Vec2){
  let mut binding = image_data.lock().unwrap();
  let texture = binding.get_or_insert_with(||
    ui.ctx().load_texture(
      name,
      load_image_from_path(path).unwrap(),
      Default::default()
    )
  );
  let size = texture.size_vec2();
  (texture.id(), size)
}

fn load_image_from_path(path: &str) -> Result<egui::ColorImage, image::ImageError> {
  let image = image::open(path).unwrap();
  let size = [image.width() as _, image.height() as _];
  let image_buffer = image.to_rgba8();
  let pixels = image_buffer.as_flat_samples();
  Ok(egui::ColorImage::from_rgba_unmultiplied(
      size,
      pixels.as_slice(),
  ))
}