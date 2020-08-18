use std::time::Instant;

mod net;
mod mnist;

fn main() {
	let train_dat = mnist::get_training_data();
	let train_lab = mnist::get_training_labels();
	let test_dat = mnist::get_testing_data();
	let test_lab = mnist::get_testing_label();

	let mut net = net::make_net(&[784,20,20,10]);
	
	println!("Test ran at {}%", test(&mut net, &test_dat, &test_lab)*100f64);

	for _i in 0..50 {		
		train(&mut net, &train_dat, &train_lab);
		println!("Test ran at {}%", test(&mut net, &test_dat, &test_lab)*100f64);
	}
}

fn test(mut net: &mut net::Net, inputs: &Vec<Vec<f64>>, outputs: &Vec<Vec<f64>>) -> f64 {
	let start = Instant::now();
	let mut ret = 0f64;

	for i in 0..inputs.len() {
		let guess = net::run_net(&mut net, &inputs[i]);
		
		let mut high: usize = 1;

		for g in 1..guess.len() {
			if guess[g] > guess[high] {
				high = g;
			}
		}

		if outputs[i][high] == 1. {
			ret += 1.;
		}
	}
	println!("Tested in {:.2?}", start.elapsed());

	return ret/(inputs.len() as f64);
}

fn train(mut net: &mut net::Net, inputs: &Vec<Vec<f64>>, outputs: &Vec<Vec<f64>>) {
	let start = Instant::now();
	for i in 0..inputs.len() {
		net::train_net(&mut net, &inputs[i], &outputs[i], i%9==0);
	}
	println!("Trained in {:.2?}", start.elapsed());
}