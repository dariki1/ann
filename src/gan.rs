use rand::random;
use crate::dnn;

pub struct GAN {
	pub discriminator: dnn::DNN,
	pub generator: dnn::DNN
}

pub fn new(discriminator_nodes: &[usize], generator_nodes: &[usize]) -> GAN {
	if discriminator_nodes[discriminator_nodes.len()-1] != 1 {
		panic!("Discriminator must have a single node for output, had {}", discriminator_nodes[discriminator_nodes.len()-1]);
	}
	if discriminator_nodes.first() != generator_nodes.last() {
		panic!("Generator output must be the same as Disciminator input");
	}
	return GAN {
		discriminator: dnn::new(discriminator_nodes),
		generator: dnn::new(generator_nodes)
	}
}

pub fn generate(mut gen: &mut dnn::DNN, noise: &Vec<f32>) -> Vec<f32> {
	return dnn::run_net(&mut gen, &noise);
}

pub fn train_discriminator(mut gan: &mut GAN, inputs: &Vec<Vec<f32>>) {
	for image in 0..inputs.len() {
		// Train on fake (generated) data
		let noise = gen_noise(gan.generator.node_vals[0].len() as u32);
		let fake = generate(&mut gan.generator, &noise);
		dnn::train_net(&mut gan.discriminator, &fake, &[0.], false);
		// Train on real data
		dnn::train_net(&mut gan.discriminator, &inputs[image], &[1.], image%5==0);
	}
}

pub fn train_generator(mut gan: &mut GAN, iters: usize) {
	for iter in 0..iters {
		let noise = gen_noise(gan.generator.node_vals[0].len() as u32);
		let fake = generate(&mut gan.generator, &noise);
		let output = dnn::run_net(&mut gan.discriminator, &fake);
		let errors = get_error(&mut gan.discriminator, output, vec![1.]);

		dnn::train_net(&mut gan.generator, &noise, &errors, iter % 10 == 9);
	}
}

/*pub fn train(mut gan: &mut GAN, inputs: &Vec<Vec<f32>>) {
	println!("Training discriminator");
	// Train Discriminator
	for image in 0..inputs.len() {
		// Train on fake (generated) data
		let noise = gen_noise(gan.generator.node_vals[0].len() as u32);
		let fake = generate(&mut gan.generator, &noise);
		dnn::train_net(&mut gan.discriminator, &fake, &[0.], false);
		// Train on real data
		dnn::train_net(&mut gan.discriminator, &inputs[image], &[1.], image%5==0);
	}

	let mut t_fake: f32 = 0.;
	let mut t_real: f32 = 0.;

	for i in 0..10 {
		t_real += dnn::run_net(&mut gan.discriminator, &inputs[i])[0];
		let noise = gen_noise(gan.generator.node_vals[0].len() as u32);
		t_fake += dnn::run_net(&mut gan.discriminator, &generate(&mut gan.generator, &noise))[0];
	}

	println!("Real average guess; {}\nFake average guess; {}", t_real/10f32, t_fake/10f32);

	println!("Training generator");
	// Train Generator
	for iter in 0..inputs.len()*2 {
		let noise = gen_noise(gan.generator.node_vals[0].len() as u32);
		let fake = generate(&mut gan.generator, &noise);
		let output = dnn::run_net(&mut gan.discriminator, &fake);
		let errors = get_error(&mut gan.discriminator, output, vec![1.]);

		dnn::train_net(&mut gan.generator, &noise, &errors, iter % 10 == 9);
	}

	t_fake = 0.;
	t_real = 0.;

	for i in 0..10 {
		t_real += dnn::run_net(&mut gan.discriminator, &inputs[i])[0];
		let noise = gen_noise(gan.generator.node_vals[0].len() as u32);
		t_fake += dnn::run_net(&mut gan.discriminator, &generate(&mut gan.generator, &noise))[0];
	}

	println!("Real average guess; {}\nFake average guess; {}", t_real/10f32, t_fake/10f32);
}*/

fn get_error(mut discriminator: &mut dnn::DNN, inputs: Vec<f32>, outputs: Vec<f32>) -> Vec<f32>{
	dnn::run_net(&mut discriminator, &inputs);

	let mut expected_outputs: Vec<f32> = outputs.clone().to_vec();

	for layer in (1..discriminator.node_vals.len()).rev() {
		let mut next_expected_values: Vec<f32> = vec![0f32; discriminator.node_vals[layer-1].len()];
		for output in 0..discriminator.node_vals[layer].len() {
			let error: f32 = expected_outputs[output] - discriminator.node_vals[layer][output].output;
			for input in 0..discriminator.node_vals[layer-1].len() {
				next_expected_values[input] += discriminator.weights[layer-1][input][output] * error;
			}
		}

		expected_outputs = next_expected_values.clone();
	}

	return expected_outputs;
}

pub fn gen_noise(noise_count: u32) -> Vec<f32> {
	let mut ret: Vec<f32> = vec![];

	for i in 0..noise_count {
		ret.push(random::<f32>());
	}

	return ret;
}