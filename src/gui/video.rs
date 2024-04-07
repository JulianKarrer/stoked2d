use std::path::Path;
use crate::*;

use ndarray::Array3;
use ocl::prm::Uchar3;
use video_rs::encode::{Encoder, Settings};
use video_rs::time::Time;


use self::gpu_version::buffers::GpuBuffers;

pub struct VideoHandler{
  encoder: Encoder,
  position: Time,
  duration: Time,
  buffer: Vec<Uchar3>
}

impl Default for VideoHandler{
  fn default() -> Self {
    video_rs::init().unwrap();
    let settings = Settings::preset_h264_yuv420p(VIDEO_SIZE.0, VIDEO_SIZE.1, false);
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let encoder =
      Encoder::new(Path::new(
        &format!("videos/{}.mp4", timestamp)
        // "test.mp4"
      ), settings).expect("failed to create encoder");
    let duration: Time = Time::from_nth_of_a_second(60);
    let position = Time::zero();
    let buffer = vec![Uchar3::new(0, 0, 0); VIDEO_SIZE.0*VIDEO_SIZE.1];
    Self { encoder, position, duration, buffer }
  }
}

impl VideoHandler{
  pub fn add_frame(&mut self, b: &GpuBuffers){
    b.image.read(&mut self.buffer).enq().unwrap();
    self.encoder.encode(
      & Array3::from_shape_fn((VIDEO_SIZE.1, VIDEO_SIZE.0, 3), |(y,x,c)|{
        self.buffer[x+y*VIDEO_SIZE.0][c]
      }), 
      &self.position
    ).expect("failed to encode frame");
    self.position = self.position.aligned_with(&self.duration).add();
  }

  pub fn finish(&mut self){
    self.encoder.finish().expect("failed to finish encoder");
  }
}

