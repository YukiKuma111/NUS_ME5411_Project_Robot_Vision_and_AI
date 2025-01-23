function net = cnnapplygrads(net, opts)
    % Updates the convolutional kernels and biases using gradient descent.
    %
    % Inputs:
    % - net: The neural network structure containing weights, biases, and gradients
    % - opts: A structure containing options for training (e.g., learning rate `alpha`)
    %
    % Outputs:
    % - net: The updated neural network structure

    % Update the weights and biases for each layer
    for l = 2 : numel(net.layers) % Start from the second layer (input layer does not require updates)
        if strcmp(net.layers{l}.type, 'c') % Check if the layer is a convolutional layer
            for j = 1 : numel(net.layers{l}.a)
                for ii = 1 : numel(net.layers{l - 1}.a)
                    % Update convolutional kernel weights
                    net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - opts.alpha * net.layers{l}.dk{ii}{j};
                end
                % Update biases
                net.layers{l}.b{j} = net.layers{l}.b{j} - opts.alpha * net.layers{l}.db{j};
            end
        end
    end

    % Update the weights and biases for the fully connected layer
    net.ffW = net.ffW - opts.alpha * net.dffW;
    net.ffb = net.ffb - opts.alpha * net.dffb;
end
