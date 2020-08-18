use std::fs;

const LABEL_OFFSET: usize = 8;
const DATA_OFFSET: usize = 16;
const IMAGE_WIDTH: usize = 28;
const IMAGE_HEIGHT: usize = 28;

pub fn get_training_data() -> Vec<Vec<f64>>{
	return read_data(&"data/train-images.idx3-ubyte");
}
pub fn get_training_labels() -> Vec<Vec<f64>>{
	return read_label(&"data/train-labels.idx1-ubyte");
}
pub fn get_testing_data() -> Vec<Vec<f64>>{
	return read_data(&"data/t10k-images.idx3-ubyte");
}
pub fn get_testing_label() -> Vec<Vec<f64>>{
	return read_label(&"data/t10k-labels.idx1-ubyte");
}

fn read_data(file_path: &str) -> Vec<Vec<f64>> {
	println!("Loading data; {}", file_path);
	let contents: Vec<u8> = fs::read(file_path).unwrap();

	let mut ret: Vec<Vec<f64>> = vec!();

	for image in 0..((contents.len()-LABEL_OFFSET)/(IMAGE_HEIGHT*IMAGE_WIDTH)) {
		ret.push(vec![0f64; IMAGE_HEIGHT*IMAGE_WIDTH]);

		for x in 0..IMAGE_WIDTH {
			for y in 0..IMAGE_HEIGHT {
				ret[image][x*IMAGE_WIDTH + y] = ((contents[image*IMAGE_HEIGHT*IMAGE_WIDTH+x*IMAGE_WIDTH+y+DATA_OFFSET] as f64/255f64) as f64);//.round();
			}
		}
	}

	return ret;
}

fn read_label(file_path: &str) -> Vec<Vec<f64>> {
	println!("Loading label; {}", file_path);
	let contents: Vec<u8> = fs::read(file_path).unwrap();

	let mut ret: Vec<Vec<f64>> = vec!();

	for image in 0..(contents.len()-LABEL_OFFSET) {
		ret.push(vec![0f64;10]);

		ret[image][contents[image+LABEL_OFFSET] as usize] = 1f64;
	}

	return ret;
}