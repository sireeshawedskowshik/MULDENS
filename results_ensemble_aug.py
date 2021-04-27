import pandas as pd
import numpy as np
for test_env in range(4):
    total_train_val_domains = 3

    out_file = 'Inevnio_validation_aug_with_train_aug/TerraIncognita_M3/env'+str(test_env)+'_with_train_aug/indiv_ensemble_no_beta/out copy.txt'
    #out_file = 'Inevnio_validation_aug_with_train_aug/TerraIncognita_M2/env'+str(test_env)+'_with_train_aug/indiv_ensemble_no_beta/out copy.txt'

    out_df= pd.read_fwf(out_file)
    out_df= out_df[out_df['epoch']<44]
    train_envs = np.setdiff1d(np.arange(4),test_env)
    train_out_split_names= ['env'+str(i)+'_out' + str(j)+'_acc' for i in train_envs for j in range(1,total_train_val_domains)]
    
    test_ensemble_name= 'env'+str(test_env)+'_in0_ens_acc'
    test_ensemble_out_name= 'env'+str(test_env)+'_out0_ens_acc'

    epochs= out_df['epoch']
    validation_acc= np.mean(np.stack([out_df[i].values for i in train_out_split_names]),axis=0)
    
    max_val_idx= np.argmax(validation_acc)
    max_val_idx_1 = np.argmax(out_df[test_ensemble_out_name].values)
    print('test_train_Domain_val','test_oracle',out_df[test_ensemble_name].iloc[max_val_idx]*100,out_df[test_ensemble_name].iloc[max_val_idx_1]*100.)