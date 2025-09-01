use crate::config::LOG_PATH;
use burn::tensor::cast::ToElement;
use burn::tensor::{Tensor, backend::Backend};
use chrono::{Utc};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

pub struct Stats {
    stopwatch: Instant,
    batch_size: usize,
    l: f32,
    log_output: String,
    f_handle: File,
}

impl Stats {
    pub fn new(batch_size: usize) -> Self {
        let f_handle = File::options()
            .create(true)
            .append(true)
            .open(LOG_PATH)
            .unwrap();
        let now = Utc::now();

        writeln!(&f_handle, "\n----{}----\n", now.format("%Y-%m-%d %H:%M:%S")).unwrap();
        Stats {
            stopwatch: Instant::now(),
            batch_size,
            l: 0.0,
            log_output: String::new(),
            f_handle,
        }
    }

    pub fn update<B: Backend>(
        &mut self,
        loss: Tensor<B, 2>,
        iteration: usize,
        name: String,
        epoch: usize,
    ) {
        let iteration = if iteration == 0 {
            return;
        } else {
            iteration
        };

        self.l += loss.clone().sum().into_scalar().to_f32();

        let elapsed = self.stopwatch.elapsed().as_secs();

        self.log_output = format!(
            "{},E:{:<6.3},I:{:<6.3},L:{:<6.3},T:{:<}m{:<}s\r",
            name,
            epoch,
            iteration * self.batch_size,
            self.l / iteration as f32,
            (elapsed / 60),
            elapsed % 60
        );

        print!("{}", &self.log_output);
        std::io::stdout().flush().unwrap();
    }

    pub fn flush(&mut self) {
        writeln!(self.f_handle, "{}", self.log_output).unwrap();
        self.stopwatch = Instant::now();
        self.l = 0.0;
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::{NdArray, ndarray::NdArrayDevice},
        tensor::Tensor,
    };
    use std::fs::{File, remove_file};
    use std::io::Read;
    use std::time::{Duration, Instant};

    type B = NdArray<f32>;

    fn cleanup_log() {
        let _ = remove_file(LOG_PATH);
    }

    #[test]
    fn test_update_formats_log_output() {
        cleanup_log();
        let device = &NdArrayDevice::default();
        let mut stats = Stats::new(4);

        // Tensor of shape [1,4] filled with ones → sum = 4.0
        let loss: Tensor<B, 2> = Tensor::ones([1, 4], device);

        // Call update with iteration = 2 and epoch = 3
        stats.update(loss.clone(), 2, "Valid".into(), 3);

        // Expected: iteration * batch_size = 8, avg loss = 4 / 2 = 2.0
        assert!(stats.log_output.contains("Valid"));
        assert!(stats.log_output.contains("E:3"));
        assert!(stats.log_output.contains("I:8"));
        assert!(stats.log_output.contains("L:2"));
        cleanup_log();
    }

    #[test]
    fn test_update_ignores_iteration_zero() {
        cleanup_log();
        let device = &NdArrayDevice::default();
        let mut stats = Stats::new(4);

        let loss: Tensor<B, 2> = Tensor::ones([1, 4], device);
        stats.update(loss, 0, "Train".into(), 1);

        // No log output should have been recorded
        assert_eq!(stats.log_output, "");
        cleanup_log();
    }

    #[test]
    fn test_flush_writes_log_and_resets_stopwatch() {
        cleanup_log();
        let mut stats = Stats::new(4);

        stats.log_output = "some log line".to_string();
        // Simulate stopwatch with a large elapsed time
        stats.stopwatch = Instant::now() - Duration::from_secs(100);

        stats.flush();

        // Verify that log_output was written to the file
        let mut contents = String::new();
        File::open(LOG_PATH)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        assert!(contents.contains("some log line"));

        // Stopwatch should be near zero again
        assert!(stats.stopwatch.elapsed().as_secs() < 2);
        cleanup_log();
    }
}

