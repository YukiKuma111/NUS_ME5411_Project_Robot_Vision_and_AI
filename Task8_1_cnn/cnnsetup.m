function net = cnnsetup(net, x, y)
    inputmaps = 1;
    % Get input feature map dimensions
    mapsize = size(squeeze(x(:, :, 1)));

    for l = 1 : numel(net.layers)   % Loop through each layer
        if strcmp(net.layers{l}.type, 's')
            % Update feature map size based on the pooling layer's scale factor
            mapsize = mapsize / net.layers{l}.scale;
            % Ensure the feature map size is an integer
            assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
            for j = 1 : inputmaps
                % Initialize the biases of the pooling layer to 0
                net.layers{l}.b{j} = 0;
            end
        end
        if strcmp(net.layers{l}.type, 'c')
            % Update feature map size based on the kernel size in the convolutional layer
            mapsize = mapsize - net.layers{l}.kernelsize + 1;
            % Calculate the fan-out value for weight initialization
            fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;
            for j = 1 : net.layers{l}.outputmaps  % Iterate over output maps
                % Calculate the fan-in value based on the number of input maps and kernel size
                fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;
                for i = 1 : inputmaps  % Iterate over input maps
                    % Initialize the convolutional kernels using Xavier initialization
                    net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                end
                % Initialize the biases of the convolutional layer to 0
                net.layers{l}.b{j} = 0;
            end
            % Update the number of input maps for the next layer
            inputmaps = net.layers{l}.outputmaps;
        end
    end
    
    % 'onum' is the number of labels; the output of the network will have one neuron per label.
    % 'fvnum' is the number of output neurons at the last feature extraction layer (fully connected).
    % 'ffb' is the biases for the neurons in the fully connected layer.
    % 'ffW' is the weights between the last feature extraction layer and the output layer.
    % Note: The last feature extraction layer is fully connected to the output layer.

    % Compute the number of neurons in the last feature extraction layer
    fvnum = prod(mapsize) * inputmaps;
    % 'onum' is the number of output neurons, equal to the number of labels
    onum = size(y, 1);
    % disp(onum)
    % Initialize biases and weights for the fully connected layer
    net.ffb = zeros(onum, 1);
    net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
end
