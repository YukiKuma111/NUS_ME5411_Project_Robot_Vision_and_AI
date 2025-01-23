function net = cnnff(net, x)
    % Forward propagation for Convolutional Neural Network
    %
    % Inputs:
    % - net: Neural network structure
    % - x: Input data
    %
    % Outputs:
    % - net: Updated network structure with forward-propagated activations

    n = numel(net.layers); % Number of layers in the network
    net.layers{1}.a{1} = x; % Assign input to the first layer
    inputmaps = 1; % Number of input maps (initially 1 for single input)

    % Forward propagation through each layer
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c') % Convolutional layer
            % For each output map in the current layer
            for j = 1 : net.layers{l}.outputmaps
                % Initialize zero matrix for convolution result
                z = zeros(size(net.layers{l - 1}.a{1}) - ...
                    [net.layers{l}.kernelsize - 1, net.layers{l}.kernelsize - 1, 0]);
                % Accumulate results of convolutions with all input maps
                for i = 1 : inputmaps
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                end
                % Apply activation function (sigmoid)
                net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});
            end
            inputmaps = net.layers{l}.outputmaps; % Update input maps for the next layer
        elseif strcmp(net.layers{l}.type, 's') % Subsampling (pooling) layer
            % Downsample each input map
            for j = 1 : inputmaps
                z = convn(net.layers{l - 1}.a{j}, ...
                    ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid'); 
                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, ...
                                       1 : net.layers{l}.scale : end, :);
            end
        end
    end

    % Flatten the activations of the last layer into a feature vector
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a)
        sa = size(net.layers{n}.a{j});
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end

    % Fully connected layer forward pass
    % Apply sigmoid activation to the weighted sum of feature vector and biases
    net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));
end
