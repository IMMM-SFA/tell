import tell
import time

data_dir = 'C:/Users/mcgr323/OneDrive - PNNL/Desktop/WRF_CSV_Files/CSV_Files'
output_dir = 'C:/Users/mcgr323/OneDrive - PNNL/Desktop/WRF_predict_output'
batch_run = True
target_ba_list = None
generate_plots = True

t0 = time.time()

df = tell.predict(data_dir=data_dir,
                  out_dir=output_dir,
                  batch_run=batch_run,
                  target_ba_list=target_ba_list,
                  generate_plots=generate_plots)

print(df)