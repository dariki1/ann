use std::time::Instant;

mod dnn;
mod gan;
mod mnist;

fn main() {
	//gan_run();
	mnist_run();
}

fn gan_run() {
	let mut gan = gan::new(&[784,1], &[10,20,20,784]);
	let train_dat = mnist::get_training_data();

	for i in 0..100 {
		let mut score: f32 = 0.;
		while score < 0.6 {
			println!("Training discriminator");
			score = 0.;
			gan::train_discriminator(&mut gan, &train_dat);
			for e in 0..100 {
				score += dnn::run_net(&mut gan.discriminator, &train_dat[e])[0];
				let noise = gan::gen_noise(gan.generator.node_vals[0].len() as u32);
				score += 1f32-dnn::run_net(&mut gan.discriminator, &gan::generate(&mut gan.generator, &noise))[0];
			}
			score /= 200f32;
			println!("Discriminator scored {}%", score*100f32);
		}

		score = 0f32;
		println!("Training Generator");
		gan::train_generator(&mut gan, 1000);
		for e in 0..100 {
			score += dnn::run_net(&mut gan.discriminator, &train_dat[e])[0];
			let noise = gan::gen_noise(gan.generator.node_vals[0].len() as u32);
			score += 1f32-dnn::run_net(&mut gan.discriminator, &gan::generate(&mut gan.generator, &noise))[0];
		}
		score /= 200f32;
		println!("Discriminator scored {}%", score*100f32);

		println!("Training done");
		gan_out(&mut gan.generator);
	}

}

fn gan_out(mut gen: &mut dnn::DNN) {
	println!();
	let output = dnn::run_net(&mut gen, &gan::gen_noise(10));
	for x in 0..28 {
		for y in 0..28 {
			print!("{}", output[x*28+y].round());
		}
		println!();
	}
	println!();
}

fn mnist_run() {
	let train_dat = mnist::get_training_data();
	let train_lab = mnist::get_training_labels();
	let test_dat = mnist::get_testing_data();
	let test_lab = mnist::get_testing_label();

	let mut net = dnn::new(&[784,100,20,10]);
	
	println!("Test ran at {}%", mnist_test(&mut net, &test_dat, &test_lab)*100f32);

	for i in 0..500 {
		println!("Loop #{}", i);
		mnist_train(&mut net, &train_dat, &train_lab);
		println!("Test ran at {}%", mnist_test(&mut net, &test_dat, &test_lab)*100f32);
	}
}

fn mnist_test(mut net: &mut dnn::DNN, inputs: &Vec<Vec<f32>>, outputs: &Vec<Vec<f32>>) -> f32 {
	let start = Instant::now();
	let mut ret = 0f32;

	for i in 0..inputs.len() {
		let guess = dnn::run_net(&mut net, &inputs[i]);
		
		let mut high: usize = 0;

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

	return ret/(inputs.len() as f32);
}

fn mnist_train(mut net: &mut dnn::DNN, inputs: &Vec<Vec<f32>>, outputs: &Vec<Vec<f32>>) {
	let start = Instant::now();
	for i in 0..inputs.len() {
		dnn::train_net(&mut net, &inputs[i], &outputs[i], i%10==9);
	}
	println!("Trained in {:.2?}", start.elapsed());
}