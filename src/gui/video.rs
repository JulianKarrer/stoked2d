use crate::*;
use std::fs::File;
use std::path::Path;

use image::codecs::jpeg::JpegEncoder;
use image::RgbImage;
use ndarray::Array3;
use ocl::prm::Uchar3;
use video_rs::encode::{Encoder, Settings};
use video_rs::time::Time;

use self::gpu_version::buffers::GpuBuffers;
use self::utils::get_timestamp;

pub struct VideoHandler {
    encoder: Encoder,
    position: Time,
    duration: Time,
    buffer: Vec<Uchar3>,
    x_size: usize,
    y_size: usize,
}

impl VideoHandler {
    pub fn new(x_size: usize, y_size: usize) -> Self {
        video_rs::init().unwrap();
        let settings = Settings::preset_h264_yuv420p(x_size, y_size, false);
        let timestamp = get_timestamp();
        let encoder = Encoder::new(
            Path::new(
                &format!("videos/{}.mp4", timestamp), // "test.mp4"
            ),
            settings,
        )
        .expect("failed to create encoder");
        let duration: Time = Time::from_nth_of_a_second(60);
        let position = Time::zero();
        let buffer = vec![Uchar3::new(0, 0, 0); x_size * y_size];
        Self {
            encoder,
            position,
            duration,
            buffer,
            x_size,
            y_size,
        }
    }
}

impl VideoHandler {
    pub fn add_gpu_frame(&mut self, b: &GpuBuffers) {
        b.image.read(&mut self.buffer).enq().unwrap();
        self.encoder
            .encode(
                &Array3::from_shape_fn((self.y_size, self.x_size, 3), |(y, x, c)| {
                    self.buffer[x + y * self.x_size][c]
                }),
                &self.position,
            )
            .expect("failed to encode frame");
        self.position = self.position.aligned_with(&self.duration).add();
    }

    pub fn add_frame(&mut self, frame: &std::vec::Vec<u8>) {
        self.encoder
            .encode(
                &Array3::from_shape_fn((self.y_size, self.x_size, 3), |(y, x, c)| {
                    frame[x * 3 + y * self.x_size * 3 + c]
                }),
                &self.position,
            )
            .expect("failed to encode frame");
        self.position = self.position.aligned_with(&self.duration).add();
    }

    pub fn add_raw_frame(
        frame: &std::vec::Vec<u8>,
        frame_number: usize,
        timestamp: u64,
        size: (usize, usize),
    ) {
        std::fs::create_dir_all(format!("videos/{}", timestamp)).unwrap();
        let mut file =
            File::create(format!("videos/{}/{:0>5}.jpg", timestamp, frame_number)).unwrap();
        let encoder = JpegEncoder::new_with_quality(&mut file, 100);
        RgbImage::from_raw(size.0 as u32, size.1 as u32, frame.to_vec())
            .unwrap()
            .write_with_encoder(encoder)
            .unwrap();
    }

    pub fn finish(&mut self) {
        self.encoder.finish().expect("failed to finish encoder");
    }
}
