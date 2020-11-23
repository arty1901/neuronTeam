import tensorflow as tf
import numpy as np
import os, time
import pickle

from cyclegan import CycleGAN
from cycle_began import CycleBeGAN
from preprocess import *

from config import *

def train(input_A, input_B):

    # Config Since it was loaded from a file, an error occurred when assigning it, so it was set as a global variable.
    global k_t_A, k_t_B, lambda_k_A, lambda_k_B, balance_A, balance_B, checkpoint_every
    global identity_lambda, generator_lr, discriminator_lr

    # Make Directory
    if os.path.exists(log_dir) is False :
        os.mkdir(log_dir)
    if os.path.exists(model_dir) is False :
        os.mkdir(model_dir)

    # Preprocessing datasets
    # If there is, run it as existing
    if os.path.exists(os.path.join("./data", "A_norm.pickle")):
        print('Preprocessed pickle file load!!!!!! \nIf you are going to train with a new file, delete the pickle files in the data.')
        with open(os.path.join("./data", "A_norm.pickle"), "rb") as fp:   # Unpickling
            A_norm = pickle.load(fp)
        with open(os.path.join("./data", "B_norm.pickle"), "rb") as fp:   # Unpickling
            B_norm = pickle.load(fp)
    else:
        A_norm, B_norm = preprocess(input_A,input_B)

    # Load model CycleBeGAN
    if began == True:
        model = CycleBeGAN(num_features = n_features, log_dir = log_dir)
    elif began == False:
        model = CycleGAN(num_features=n_features, g_type=g_type, log_dir=log_dir)

    # Load Saved Model Checkpoint
    try:
        saved_global_step = model.load(model_dir)
        if saved_global_step is None:
            # check_point If none, start from 0
            saved_global_step = -1

    # Error when loading incorrectly saved one
    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise
        
    print("Start Training...")
    try:
        # Start training
        for epoch in range(saved_global_step + 1, n_epochs) :
            print("Epoch : %d " % epoch ) 
            start_time = time.time()
            train_A, train_B = sample_train_data(dataset_A=A_norm, dataset_B=B_norm,n_frames=n_frames) # random data
        
            n_samples = train_A.shape[0]

            # Cycle beGAN
            if began == True:
                # First I don't know what it is
                for i in range(n_samples) : # mini_ batch_size = 1
                    n_iter = n_samples * epoch + i
                    if n_iter % 50 == 0:
                        
                        k_t_A = k_t_A + (lambda_k_A *balance_A)
                        if k_t_A > 1:
                            k_t_A = 1
                        if k_t_A < 0 :
                            k_t_A = 0
                        
                        k_t_B = k_t_B + (lambda_k_B *balance_B)
                        if k_t_B > 1.0:
                            k_t_B = 1.0
                        if k_t_B < 0. :
                            k_t_B = 0.
                
                    if n_iter > 10000 :
                        identity_lambda = 0
                    if n_iter > 200000 :
                        generator_lr = max(0, generator_lr - generator_lr_decay)
                        discriminator_lr = max(0, discriminator_lr - discriminator_lr_decay)
                
                    start = i
                    end = start + 1
                    # Loss exoneration
                    generator_loss, discriminator_loss, measure_A, measure_B, k_t_A, k_t_B, balance_A, balance_B = model.train(
                                    input_A=train_A[start:end], input_B=train_B[start:end], 
                                    lambda_cycle=lambda_cycle,
                                    lambda_identity=lambda_identity,
                                    gamma_A=gamma_A, gamma_B=gamma_B, lambda_k_A=lambda_k_A, lambda_k_B=lambda_k_B,
                                    generator_learning_rate=generator_learning_rate,
                                    discriminator_learning_rate=discriminator_learning_rate, 
                                    k_t_A = k_t_A, k_t_B = k_t_B)
            # CycleGAN
            elif began == False:
                for i in range(n_samples) : # mini_ batch_size = 1
                    n_iter = n_samples * epoch + i
                    
                    # Began adds k_t_A in this part
                    
                    if n_iter > 10000 :
                        identity_lambda = 0
                    if n_iter > 200000 :
                        generator_lr = max(0, generator_lr - generator_lr_decay)
                        discriminator_lr = max(0, discriminator_lr - discriminator_lr_decay)
                    
                    start = i
                    end = start + 1
                    
                    # Ross type is also different
                    generator_loss, discriminator_loss = model.train(input_A = train_A[start:end], 
                                                                    input_B = train_B[start:end], 
                                                                    cycle_lambda = cycle_lambda,
                                                                    identity_lambda = identity_lambda,
                                                                    generator_lr = generator_lr,
                                                                    discriminator_lr = discriminator_lr)
            
            
            end_time = time.time()
            epoch_time = end_time-start_time
            print("Generator Loss : %f, Discriminator Loss : %f, Time : %02d:%02d" % (generator_loss, discriminator_loss,(epoch_time % 3600 // 60),(epoch_time % 60 // 1)))

            
            # every Save per epoch
            if epoch % checkpoint_every == 0:
                model.save(directory = model_dir, filename = "model", epoch=epoch)
                print(epoch, 'model save complete')
    
    finally:
        print('Save model with incorrect termination or training ended')
        model.save(directory = model_dir, filename = "model", epoch=epoch)


if __name__ == "__main__" :
    train(input_A = dataset_A, input_B = dataset_B)
    print("Training Done!")


# 코랩 런타임 세션 종료 방지
# f12 눌러 console창에 주석풀고 입력해주세요~!
# function ClickConnect() {
#     // 백엔드를 할당하지 못했습니다.
#     // GPU이(가) 있는 백엔드를 사용할 수 없습니다. 가속기가 없는 런타임을 사용하시겠습니까?
#     // 취소 버튼을 찾아서 클릭 
#     var buttons = document.querySelectorAll("colab-dialog.yes-no-dialog paper-button#cancel");
#     buttons.forEach(function(btn) { btn.click(); });
#     console.log("1분마다 자동 재연결");
#     document.querySelector("colab-toolbar-button#connect").click();
# }
# setInterval(ClickConnect,1000*60);

# 출처: https://bryan7.tistory.com/1077 [민서네집]
