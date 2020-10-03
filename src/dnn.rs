use rand::random;

pub struct Node {
	value: f32,
	pub output: f32,
}

pub struct DNN {
	pub weights: Vec<Vec<Vec<f32>>>,
	weights_adjustment: Vec<Vec<Vec<f32>>>,
	pub node_vals: Vec<Vec<Node>>,
	bias_per_layer: usize,
	bias_output: f32,
	pub learn_rate: f32
}

pub fn new(nodes_per_layer: &[usize]) -> DNN{
	let mut net: DNN = DNN {
		weights: vec![vec![]; nodes_per_layer.len()-1],
		weights_adjustment: vec![vec![]; nodes_per_layer.len()-1],
		node_vals: vec![],
		bias_per_layer: 1,
		bias_output: activation_function(1f32),
		learn_rate: 0.1
	};

	for layer in 0..net.weights.len() {
		net.weights[layer] = vec![vec![]; nodes_per_layer[layer]+net.bias_per_layer];
		net.weights_adjustment[layer] = vec![vec![]; nodes_per_layer[layer]+net.bias_per_layer];
		net.node_vals.push(vec![]);
		for input in 0..net.weights[layer].len() {
			net.weights[layer][input] = vec![0f32; nodes_per_layer[layer+1]];
			net.weights_adjustment[layer][input] = vec![0f32; nodes_per_layer[layer+1]];
			if input < nodes_per_layer[layer] {
				net.node_vals[layer].push(Node {
					value: 0f32,
					output: activation_function(0f32),
				});
			}
			for output in 0..net.weights[layer][input].len() {
				net.weights[layer][input][output] = random::<f32>()*2f32-1f32;
			}
		}
	}

	net.node_vals.push(vec![]);

	for _output in 0..nodes_per_layer[nodes_per_layer.len()-1] {
		net.node_vals[nodes_per_layer.len()-1].push(Node {
			value: 0f32,
			output: activation_function(0f32),
		});
	}

	return net;
}

pub fn run_net(net: &mut DNN, inputs: &[f32]) -> Vec<f32> {
	for input in 0..inputs.len() {
		net.node_vals[0][input].value = inputs[input];
		net.node_vals[0][input].output = inputs[input];
	}

	for layer in 1..net.node_vals.len() {		
		for output in 0..net.node_vals[layer].len() {
			net.node_vals[layer][output].value = 0f32;
			for input in 0..net.weights[layer-1].len() {
				net.node_vals[layer][output].value += net.weights[layer-1][input][output] * if input < net.node_vals[layer-1].len() {
					net.node_vals[layer-1][input].output
				} else {
					net.bias_output
				};
			}
			net.node_vals[layer][output].output = activation_function(net.node_vals[layer][output].value);
		}
	}

	let mut ret: Vec<f32> = vec!();
	for node in 0..net.node_vals[net.node_vals.len()-1].len() {
		ret.push(net.node_vals[net.node_vals.len()-1][node].output);
	}

	return ret;
}

pub fn train_net(mut net: &mut DNN, inputs: &[f32], outputs: &[f32], execute: bool) {
	run_net(&mut net, &inputs);

	let mut expected_outputs: Vec<f32> = outputs.clone().to_vec();

	for layer in (1..net.node_vals.len()).rev() {
		let mut next_expected_values: Vec<f32> = vec![0f32; net.node_vals[layer-1].len()];

		for output in 0..net.node_vals[layer].len() {
			let error: f32 = expected_outputs[output] - net.node_vals[layer][output].output;
			let gradient: f32 = activation_derivative(net.node_vals[layer][output].value) * error * net.learn_rate;

			for input in 0..net.weights[layer-1].len() {
				net.weights_adjustment[layer-1][input][output] += gradient * if input < net.node_vals[layer-1].len() {
					net.node_vals[layer-1][input].output
				} else {
					net.bias_output
				};

				if input < next_expected_values.len() {
					next_expected_values[input] += net.weights[layer-1][input][output] * error;
				}

				if execute {
					net.weights[layer-1][input][output] += net.weights_adjustment[layer-1][input][output];
					net.weights_adjustment[layer-1][input][output] = 0f32;
				}
			}
		}

		expected_outputs = next_expected_values.clone();		
	}
}


fn activation_function(value: f32) -> f32 {
	sigmoid(value)
}

fn activation_derivative(value: f32) -> f32 {
	sigmoid_derivative(value)
}

/* Tends to turn values into NaN, I'll fix it later
fn tanh(value: f32) -> f32 {
	return 2f32*sigmoid(2f32*value)-1f32;
}

fn tanh_derivative(value: f32) -> f32 {
	return 4f32*sigmoid_derivative(2f32*value);
}
*/
fn sigmoid(value: f32) -> f32 {
	return 1f32/(1f32+(-value).exp());
}

fn sigmoid_derivative(value: f32) -> f32 {	
	return sigmoid(value)*(1f32-sigmoid(value));
}