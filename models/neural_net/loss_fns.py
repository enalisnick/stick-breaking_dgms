import theano.tensor as T

def calc_binaryVal_negative_log_likelihood(data, probabilities, axis_to_sum=1):
	if axis_to_sum != 1:
            # addresses the case where we marginalize                                                                                                           
            data = T.extra_ops.repeat(T.shape_padaxis(data, axis=1), repeats = probabilities.shape[1], axis=1)
        return - T.sum(data * T.log(probabilities) + (1 - data) * T.log(1 - probabilities), axis=axis_to_sum)

def calc_categoricalVal_negative_log_likelihood(data, probabilities, axis_to_sum=1):
	if axis_to_sum != 1:
            # addresses the case where we marginalize                                                                                                                                                                    
            data = T.extra_ops.repeat(T.shape_padaxis(data, axis=1), repeats = probabilities.shape[1], axis=1)
        return - T.sum(data * T.log(probabilities), axis=axis_to_sum)

def calc_realVal_negative_log_likelihood(data, recon, axis_to_sum=1):
	if axis_to_sum != 1:
		# addresses the case where we marginalize                 
		data = T.extra_ops.repeat(T.shape_padaxis(data, axis=1), repeats = recon.shape[1], axis=1)
	return .5 * T.sum( (data - recon)**2, axis=axis_to_sum )

def calc_poissonVal_negative_log_likelihood(data, recon, axis_to_sum=1):
	if axis_to_sum != 1:
		# addresses the case where we marginalize                                              
		data = T.extra_ops.repeat(T.shape_padaxis(data, axis=1), repeats = recon.shape[1], axis=1)
	return T.sum( T.exp(recon) - data * recon, axis=axis_to_sum )

def calc_cat_entropy(probabilities):
	return - T.sum(probabilities * T.log(probabilities), axis=1)

def calc_cat_kl_divergence(p1, p2):
	return -T.sum(p1 * T.log(p2), axis=1) - calc_cat_entropy(p1)

def calc_prediction_errors(class_idxs, pred_idxs):
	return T.sum(T.neq(class_idxs, pred_idxs))
