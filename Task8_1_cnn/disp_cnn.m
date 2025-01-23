function disp_cnn(cnn)
    % Display the structure of the CNN network.
    disp('CNN Structure:');
    
    % Iterate through each layer in the CNN
    for i = 1:length(cnn.layers)
        % Get the current layer
        layer = cnn.layers{i};
        
        % Display the layer number
        disp(['Layer ' num2str(i) ':']);
        
        % Display the type of the current layer
        disp(['  Type: ' layer.type]);
        
        % Display the output maps if the field exists
        if isfield(layer, 'outputmaps')
            disp(['  Output Maps: ' num2str(layer.outputmaps)]);
        end
        
        % Display the kernel size if the field exists
        if isfield(layer, 'kernelsize')
            disp(['  Kernel Size: ' num2str(layer.kernelsize)]);
        end
        
        % Display the scale factor if the field exists
        if isfield(layer, 'scale')
            disp(['  Scale: ' num2str(layer.scale)]);
        end
        
        % Add a blank line for readability
        disp(' ');
    end
end
