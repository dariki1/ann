use rand::random;

struct Node {
	value: f64,
	output: f64,
}

pub struct Net {
	pub weights: Vec<Vec<Vec<f64>>>,
	weights_adjustment: Vec<Vec<Vec<f64>>>,
	node_vals: Vec<Vec<Node>>,
	bias_per_layer: usize,
	bias_output: f64,
	learn_rate: f64
}

pub fn make_net(nodes_per_layer: &[usize]) -> Net{
	let mut net: Net = Net {
		weights: vec![vec![]; nodes_per_layer.len()-1],
		weights_adjustment: vec![vec![]; nodes_per_layer.len()-1],
		node_vals: vec![],
		bias_per_layer: 1,
		bias_output: 1f64,
		learn_rate: 0.1
	};

	for layer in 0..net.weights.len() {
		net.weights[layer] = vec![vec![]; nodes_per_layer[layer]+net.bias_per_layer];
		net.weights_adjustment[layer] = vec![vec![]; nodes_per_layer[layer]+net.bias_per_layer];
		net.node_vals.push(vec![]);
		for input in 0..net.weights[layer].len() {
			net.weights[layer][input] = vec![0f64; nodes_per_layer[layer+1]];
			net.weights_adjustment[layer][input] = vec![0f64; nodes_per_layer[layer+1]];
			if input < nodes_per_layer[layer] {
				net.node_vals[layer].push(Node {
					value: 0f64,
					output: sigmoid(0f64),
				});
			}
			for output in 0..net.weights[layer][input].len() {
				net.weights[layer][input][output] = random::<f64>()*2f64-1f64;
			}
		}
	}

	net.node_vals.push(vec![]);

	for _output in 0..nodes_per_layer[nodes_per_layer.len()-1] {
		net.node_vals[nodes_per_layer.len()-1].push(Node {
			value: 0f64,
			output: sigmoid(0f64),
		});
	}

	return net;
}

pub fn run_net(net: &mut Net, inputs: &[f64]) -> Vec<f64> {
	for input in 0..inputs.len() {
		net.node_vals[0][input].value = inputs[input];
		net.node_vals[0][input].output = inputs[input];
	}

	for layer in 1..net.node_vals.len() {		
		for output in 0..net.node_vals[layer].len() {
			net.node_vals[layer][output].value = 0f64;
			for input in 0..net.weights[layer-1].len() {
				net.node_vals[layer][output].value += net.weights[layer-1][input][output] * if input < net.node_vals[layer-1].len() {
					net.node_vals[layer-1][input].output
				} else {
					sigmoid(net.bias_output)
				};
			}
			net.node_vals[layer][output].output = sigmoid(net.node_vals[layer][output].value);
		}
	}

	let mut ret: Vec<f64> = vec!();
	for node in 0..net.node_vals[net.node_vals.len()-1].len() {
		ret.push(net.node_vals[net.node_vals.len()-1][node].output);
	}

	return ret;
}

pub fn train_net(mut net: &mut Net, inputs: &[f64], outputs: &[f64], execute: bool) {
	run_net(&mut net, &inputs);

	let mut expected_outputs: Vec<f64> = vec!();

	for output in 0..outputs.len() {
		expected_outputs.push(outputs[output]);
	}

	for layer in (1..net.node_vals.len()).rev() {
		let mut next_expected_values: Vec<f64> = vec![0f64; net.node_vals[layer-1].len()];

		for output in 0..net.node_vals[layer].len() {
			let error: f64 = expected_outputs[output] - net.node_vals[layer][output].output;
			let gradient: f64 = sigmoid_derivative(net.node_vals[layer][output].value) * error * net.learn_rate;

			for input in 0..net.weights[layer-1].len() {
				net.weights_adjustment[layer-1][input][output] += gradient * if input < net.node_vals[layer-1].len() {
					net.node_vals[layer-1][input].output
				} else {
					sigmoid(net.bias_output)
				};

				if input < next_expected_values.len() {
					next_expected_values[input] += net.weights[layer-1][input][output] * error;
				}

				if execute {
					net.weights[layer-1][input][output] += net.weights_adjustment[layer-1][input][output];
					net.weights_adjustment[layer-1][input][output] = 0f64;
				}
			}
		}

		expected_outputs = vec!();
		for output in 0..next_expected_values.len() {
			expected_outputs.push(next_expected_values[output]);
		}
	}
}

fn sigmoid(value: f64) -> f64 {
	return 1f64/(1f64+(-value).exp());
}

fn sigmoid_derivative(value: f64) -> f64 {
	return sigmoid(value)*(1f64-sigmoid(value));
}