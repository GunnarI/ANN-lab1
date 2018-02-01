function [missclass, timevector] = singlePerceptronLearningSeq(patterns, targets, eta, weights, epoch, plotId)

max_iter = epoch;     %Max number of iterations
iter = 0;           %iteration counter

out = zeros(1,length(patterns));

h = animatedline;

timevector = zeros(1,max_iter);
tstart = tic;
while iter <= max_iter
    iter = iter + 1;
    
    for k = 1:length(patterns)
        % forward pass
        out(k) = weights * patterns(:,k);   %Bias part is included in both the weights and the data

        %backward pass
        delta_out = (targets(k) - sign(out(k)));

        %weight update
        weights(1,1) = weights(1,1) + eta.*delta_out*patterns(1,k)';
        weights(1,2) = weights(1,2) + eta.*delta_out*patterns(2,k)';
        weights(1,3) = weights(1,3) + eta.*delta_out*patterns(3,k)';

    end
    if plotId
        %plotting data with bias
        data_weights = weights(1,1:2);
        threshold = -weights(1,3)/(data_weights*data_weights');      %normalised bias is threshold
        norm_weights = sqrt(data_weights*data_weights');

        %Plotting of weights
        hold on
        clearpoints(h);
        x = 2.*[weights(1),weights(1)];
        y = 2.*[weights(2),weights(2)];
        x2 = 2.*[-weights(2),weights(2)];
        y2 = 2.*[weights(1),-weights(1)];
        xpoints = x*threshold + x2/norm_weights;
        ypoints = y*threshold + y2/norm_weights;
        addpoints(h,xpoints(1),ypoints(1));
        addpoints(h,xpoints(2),ypoints(2));
        drawnow;
    else
        missclass = 0;
        for i = 1:length(targets)
            if targets(i) > 0 && out(i) < 0
                missclass = missclass + 1;
            elseif targets(i) < 0 && out(i) > 0
                missclass = missclass + 1;
            end
        end
        addpoints(h,iter-1,missclass);
        drawnow;
    end

    timevector(iter) = toc(tstart);
    %pause(0.1)

end
h.Color = 'red';
hold off

% Show misclassifications
missclass = 0;
for i = 1:length(targets)
    if targets(i) > 0 && out(i) < 0
        missclass = missclass + 1;
    elseif targets(i) < 0 && out(i) > 0
        missclass = missclass + 1;
    end
end

end