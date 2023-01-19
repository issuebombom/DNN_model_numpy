import numpy as np

class Initializers:
    """Initializers
    
    Initializer list:
        xavier_normal
        he_normal
        uniform
    """
    
    def __init__(self, fn_name='xavier_normal'):
        self.func = getattr(self, fn_name)

    
    def __str__(self):
        return f"Initialization Function: {self.func.__name__}"

    
    def initialize(self, model, seed=None):
        """initialize weights & biases
        """
        
        np.random.seed(seed)
        parameters = dict()
        layers = len(model)
        
        for L in range(1, layers): # 1, 2, 3        
            weights, biases = self.func(model[L-1], model[L]) # weights & biases
            parameters["W" + str(L)] = weights
            parameters["b" + str(L)] = biases

        return parameters

    # randn은 기본적으로 N(0, 1)이지만 분포를 변경하고 싶다면 E + STD * randn을 하면 된다.
    def xavier_normal(self, input_n_neurons, output_n_neurons):
        """Xavier_initializer
        """

        weights = np.random.randn(input_n_neurons, output_n_neurons) * np.sqrt(1/input_n_neurons)
        biases = np.zeros(output_n_neurons)
        
        return (weights, biases)

    def he_normal(self, input_n_neurons, output_n_neurons):
        """HE initializer
        """

        weights = np.random.randn(input_n_neurons, output_n_neurons) * np.sqrt(2/input_n_neurons)
        biases = np.zeros(output_n_neurons)

        return (weights, biases)

    def uniform(self, input_n_neurons, output_n_neurons):
        """Uniform initializer
        """

        weights = np.random.uniform(-np.sqrt(6/input_n_neurons), np.sqrt(6/input_n_neurons), size=(input_n_neurons, output_n_neurons))
        biases = np.zeros(output_n_neurons)

        return (weights, biases)


class ActivationFunction:
    """Activation Function
    
    Activation list:
        leaky_relu
        relu
        sigmoid
        softmax
    """
    
    def __init__(self, fn_name:str, derivative=False):
        self.func = getattr(self, fn_name)
        self.derivative = derivative
        
    def __call__(self, *args):        
        return self.func(*args)    

    def __str__(self):
        return f"Activation Function: {self.func.__name__}"

    
    def sigmoid(self, x):
        """Sigmoid
        """
        
        if self.derivative():
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        
        return 1/(1 + np.exp(-x))
    
    def relu(self, x):
        """ReLU
        """
        
        if self.derivative():
            def f(x):
                return 0 if x < 0 else 1
            f = np.vectorize(f)
            return f(x)
        
        return np.maximum(0, x)

    def leaky_relu(self, x):
        """Leaky_ReLU
        """
        
        if self.derivative:
            def f(x):
                return 0.01 if x < 0 else 1
            f = np.vectorize(f)
            return f(x)
            
        return np.maximum(0.01*x, x)

    def softmax(self, x):
        """Softmax
        
        derivative_softmax의 경우 input으로 tuple(prediction, Y)을 받아야 한다.
        """
        
        if self.derivative: # Softmax에 대한 Cross Entropy 함수 미분
            prediction, Y = x # input으로 tuple(prediction, Y)을 받아야 한다.
            return prediction - Y 
        
        def f(x): # np.max는 overflow 방지용
            return np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x)))


        if np.ndim(x) == 1: # 배치 묶음이 아닐 경우 인위적으로 1 배치 생성
            x = np.expand_dims(x, axis=0)
            return np.array([f(z) for z in x]).squeeze() # 배치 차원 제거

        return np.array([f(z) for z in x])

    # softmax의 도함수(사용 보류)
    def derivative_softmax_with_loss(self, x):
        """Softmax(z)에서 z에 대한 미분

        야코비안 행렬 출력
        : 미분 결과가 야코비안 행렬이 될 경우 next step에서의 연산을 위한
        reshape를 어떤 방식으로 해야할지 모르겠다.
        """

        results = []
        for pred in x: # x = softmax(z)를 의미함

            pred_len = pred.shape[-1] # a와 z는 길이가 같다.
            single_result = np.zeros((pred_len, pred_len))

            for i in range(pred_len):
                for j in range(pred_len):

                    '''
                    if i == j:
                        single_result[i][j] = pred[j] * (1-pred[j])
                    else:
                        single_result[i][j] = -pred[i] * pred[j]
                    '''

                    delta = 1 if i == j else 0
                    single_result[i][j] = pred[i] * (delta - pred[j]) if i==j else 0

            results.append(single_result)

        return np.array(results)
    
class LossFunction:
    """Loss Function
    
    Function list:
        mse
        categorical_cross_entropy
    """
    
    def __init__(self, fn_name:str):
        self.func = getattr(self, fn_name)
    
    def __call__(self, *args):
        return self.func(*args)    

    def __str__(self):
        return f"Loss Function: {self.func.__name__}"
        

    def mse(self, val_X, val_Y, parameters, activation_fn):
        """MSE
        """

        prediction, _ = Computations.feed_forward(val_X, parameters, activation_fn)
        # (x-y)^2의 평균으로 각 row의 loss를 구한 뒤 batch_size 단위의 평균 loss 연산 
        return np.mean((prediction - val_Y) ** 2) / 2 # 미분의 편의를 위해 /2 수행

    def categorical_cross_entropy(self, val_X, val_Y, parameters, activation_fn):
        """Catgorical Cross Entoropy
        """

        prediction, _ = Computations.feed_forward(val_X, parameters, activation_fn)
        return np.mean(-np.sum(prediction * np.log(val_Y+1e-7), axis=1))


class Evaluation:
    """Evaluation

    Function List:
        accuracy
        precision
        recall
        f1_score
    """

    def __init__(self, fn_name:str):
        self.func = getattr(self, fn_name)
  
    def __call__(self, prediction_indices, target_indices):
        return self.func(prediction_indices, target_indices)    

    def __str__(self):
        return f"Evaluation Metric: {self.func.__name__}"
    
    
    def accuracy(self, prediction_indices, target_indices):
        """Accuracy
        """

        return np.mean(prediction_indices == target_indices)
    
    def precision(self, prediction_indices, target_indices):
        """Precision
        """
        
        # precision
        labels = np.unique(prediction_indices)
        score_by_label = np.zeros((len(labels)))
        
        for i in labels:
            TPFP_indices = np.nonzero(prediction_indices == i) # i로 예측한 값의 위치
            TPFP = TPFP_indices[0].shape[0] # i로 예측한 값의 개수

            TP_all = prediction_indices==target_indices # 예상과 정답이 일치하는 수
            TP = np.sum(TP_all[TPFP_indices]) # TP_all 중 i값이 True 경우의 합

            score_by_label[i] = TP / TPFP
        score = np.mean(score_by_label)
        
        return score, score_by_label
    
    def recall(self, prediction_indices, target_indices):
        """Recall
        """
        
        # recall
        labels = np.unique(target_indices)
        score_by_label = np.zeros((len(labels)))
        
        for i in labels:
            TPFN_indices = np.nonzero(target_indices == i) # i가 정답인 값의 위치
            TPFN = TPFN_indices[0].shape[0]# 정답이 i인 값의 개수

            TP_all = prediction_indices==target_indices # 예상과 정답이 일치하는 수
            TP = np.sum(TP_all[TPFN_indices]) # TP_all 중 i값이 True 경우의 합

            score_by_label[i] = TP / TPFN
        score = np.mean(score_by_label)
        
        return score, score_by_label

    def f1_score(self, prediction_indices, target_indices):
        """F1 score
        """
        
        precision_score, _ = self.precision(prediction_indices, target_indices)
        recall_score, _ = self.recall(prediction_indices, target_indices)
        
        return 2 / (1 / precision_score + 1 / recall_score)


# Adam optimizer
class Optimizers:
    """Optimizers
    
    Function list:
        SGD
        Adam
    """
    
    def __init__(self, fn_name:str='Adam'):
        self.func = getattr(self, fn_name)
    
    def __call__(self, **kwargs):
        return self.func(**kwargs)

    def __str__(self):
        return f"Optimizer Function: {self.func.__name__}"
    
        
    class Adam:
        """Adam optimization algorithm
        """

        def __init__(self, lr=0.001, betas=(0.9, 0.999), epsilon=1e-08):

            self.t = 0
            self.lr = lr
            self.beta1 = betas[0]
            self.beta2 = betas[1]
            self.epsilon = epsilon


        def update(self, parameters, gradients):
            """Update Parameters with Adam
            """

            m = {key: np.zeros_like(value) for key, value in gradients.items()}
            v = m.copy()

            self.t += 1
            curr_lr = self.lr * np.sqrt(1-self.beta2 ** self.t) / (1-self.beta1 ** self.t)

            layers = len(parameters) // 2 # num of parameter layers
            for L in range(1, layers+1):

                pW = parameters["W" + str(L)]
                pb = parameters["b" + str(L)]
                gW = gradients["dW" + str(L)]
                gb = gradients["db" + str(L)]
                mW = m["dW" + str(L)]
                mb = m["db" + str(L)]
                vW = v["dW" + str(L)]
                vb = v["db" + str(L)]

                mW = (self.beta1 * mW + (1-self.beta1) * gW) / (1 - self.beta1 ** self.t)
                mb = (self.beta1 * mb + (1-self.beta1) * gb) / (1 - self.beta1 ** self.t)

                vW = (self.beta2 * vW + (1-self.beta2) * np.power(gW, 2)) / (1 - self.beta2 ** self.t)
                vb = (self.beta2 * vb + (1-self.beta2) * np.power(gb, 2)) / (1 - self.beta2 ** self.t)

                # update
                parameters["W" + str(L)] -= curr_lr * mW / (np.sqrt(vW) + self.epsilon)
                parameters["b" + str(L)] -= curr_lr * mb / (np.sqrt(vb) + self.epsilon)

            return parameters


    # SGD(mini-batch) optimizer  
    class SGD:
        """Stochastic Gradient Descent (mini-batch)
        """

        def __init__(self, lr):
            self.lr = lr

        def update(self, parameters, gradients):
            """Update Parameters with SGD
            """
            layers = len(parameters) // 2

            for L in range(1, layers+1):
                parameters["W" + str(L)] -= self.lr * gradients["dW" + str(L)]
                parameters["b" + str(L)] -= self.lr * gradients["db" + str(L)]

            return parameters


class Computations:

    def feed_forward(X, parameters, activation_fn):
        """Feed forward
        
        미니 배치를 대상으로 피드 포워드를 진행한다.
        """
        a_func = ActivationFunction(activation_fn[0])
        a_func_last = ActivationFunction(activation_fn[1])
        
        cache = {"a0": X}
        layers = len(parameters) // 2
        for L in range(1, layers+1): # 1, 2, 3
        
            prev_a = cache["a" + str(L-1)]
            W = parameters["W" + str(L)]
            b = parameters["b" + str(L)]

            Z = np.matmul(prev_a, W) + b
            
            if L == layers: # 마지막 레이어에서 softmax
                a = a_func_last(Z)
            else:
                a = a_func(Z)

            cache["Z" + str(L)] = Z
            cache["a" + str(L)] = a
        
        return a, cache


    def back_prop(prediction, Y, cache, parameters, activation_fn, loss_fn):
        """Back Propagation
        
        미니배치 단위로 피드포워드한 결과를 대상으로 backward 연산을 진행한다.
        """

        d_a_func = ActivationFunction(activation_fn[0], derivative=True)
        d_a_func_last = ActivationFunction(activation_fn[1], derivative=True)
        
        gradients = {}
        layers = len(parameters) // 2
        
        for L in range(layers, 0, -1):
            
            Z = cache["Z" + str(L)] # (bs, 10)
            W = parameters["W" + str(L)]
            a = cache["a" + str(L)]
            a_prev = cache["a" + str(L-1)]
            
            # 마지막 layer에서 activation에 대한 미분
            if L == layers:
                # Softmax에 대한 Cross Entropy 함수 미분
                if activation_fn[1] == 'softmax' and loss_fn == 'categorical_cross_entropy':
                    a = prediction, Y
                    dZ = d_a_func_last(a) # (bs, 10)
                
                # sigmoid에 대한 z 미분
                elif activation_fn[1] == 'sigmoid' and loss_fn == 'mse':
                    da = (prediction - Y) / prediction.shape[-1] # MSE에서 a에 대한 도함수 dE/da, (bs, 10)
                    dZ = da * d_a_func_last(Z)
                
                else:
                    raise Exception(
                        "You have to choose softmax-categorical_cross_entropy pair \
    or sigmoid-mse pair only"
                    )
                
            else: # leaky_relu 미분
                dZ = da * d_a_func(Z) # (bs, 10)의 곱
                
            dW = np.einsum("ba,bz->baz", a_prev, dZ) # (bs, 64)와 (bs, 10)의 외적 -> (bs, 64, 10)
            db = dZ * 1 # dZ랑 같음
            da = np.einsum("bz,wz->bw", dZ, W) # (bs, 10)과 (64, 10)의 내적 -> (bs, 64)
            
            # 배치별 기울기값에 대한 평균값을 추가한다.
            gradients["dW" + str(L)] = np.mean(dW, axis=0)
            gradients["db" + str(L)] = np.mean(db, axis=0)
        
        return gradients


    def compute_loss(val_X, val_Y, parameters, activation_fn, loss_fn):
        """MSE
        
        미니 배치 단위 평균 loss를 구한다.
        """
        l_func = LossFunction(loss_fn)
        
        return l_func(val_X, val_Y, parameters, activation_fn)


    def compute_evaluation(val_X, val_Y, parameters, activation_fn, eval_metric):
        """Compute Evaluation
        """
        
        # get prediction indices and target indices
        prediction, _ = Computations.feed_forward(val_X, parameters, activation_fn)
        prediction_indices = np.argmax(prediction, axis=1)
        target_indices = np.argmax(val_Y, axis=1)
        
        # evaluate
        evaluation_result = {eval_metric: {}}
        evaluation_metric = Evaluation(eval_metric)
        calc_result = evaluation_metric(prediction_indices, target_indices)

        if isinstance(calc_result, np.float64):
            evaluation_result[eval_metric]['score'] = calc_result

        else:
            evaluation_result[eval_metric]['score'] = calc_result[0]
            evaluation_result[eval_metric]['score_by_label'] = calc_result[1]
        
        return evaluation_result