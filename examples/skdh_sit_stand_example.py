import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from plotly import graph_objects as go
from skdh.gait import GaitSymmetryIndex, GaitLumbar
from dash import Dash, html, dcc
app = Dash(__name__)

import skdh

'''
The default pipeline is the following steps:

skdh.gaitv3.substeps.PreprocessGaitBout()
skdh.gaitv3.substeps.ApCwtGaitEvents() or skdh.gaitv3.substeps.VerticalCwtGaitEvents()
skdh.gaitv3.substeps.CreateStridesAndQc()
skdh.gaitv3.substeps.TurnDetection()

'''
np.set_printoptions(precision=20, suppress=False)

def read_custom_txt():
    file = 'EMG_squat.txt'
    file_lines = [i.split(',') for i in open(file).readlines()]
    _data = np.array([int(i) for line in file_lines for i in line if i.strip().isdigit()])
    ts = len(_data)
    print(_data.shape)
    time =  np.array(pd.date_range(start='2024-02-07-15:12:1', periods=ts, freq='ms').to_list(), dtype='datetime64[ms]')
    save_path = './skdh_emg_data.npz'
    np.savez(save_path, time=time, accel=_data)

    data = skdh.io.ReadNumpyFile(allow_pickle=True).predict(file=save_path)
    return data

def read_custom_csv():
    data_path = "./E006_walk_1.csv"
    _df = pd.read_csv(data_path, sep=',').dropna()
    print(_df.head())
    _df['time'] = pd.date_range(start='2024-02-07-15:12:1', periods=len(_df), freq='7ms').to_list()
    
    data_path_new = data_path.replace(".csv", "_date.csv") 
    _df.to_csv(data_path_new, index=False, date_format='%Y-%m-%d %H:%M:%S.%f')
    _df = pd.read_csv(data_path_new, sep=',')
    print(_df.head())
    imu = 7
    data = skdh.io.ReadCSV(time_col_name="time",
                           accel_col_names=[f"Avanti sensor {imu}: ACC.X {imu}",
                                            f"Avanti sensor {imu}: ACC.Y {imu}",
                                            f"Avanti sensor {imu}: ACC.Z {imu}"],
                           fill_gaps=True,
                        #    to_datetime_kwargs={'unit':'ms'},
                           accel_in_g=True,
                           g_value=9.81,
                           read_csv_kwargs={'sep':',', 'header':0},
                           ).predict(file=data_path_new)
    return data

def read_custom_npz():
    data_path = "Charles_imu_20231204115238_SAS_0.npz"
    data = np.load(data_path, allow_pickle=True)
    print('old_shape: ', data['imu1'].shape)
    _data = np.repeat(data['imu1'], repeats=10, axis=1).T
    ts, ch = _data.shape
    print('new_shape: ', _data.shape)
    time =  np.array(pd.date_range(start='2024-02-07-15:12:1', periods=ts, freq='7ms').to_list(), dtype='datetime64[ms]')
    save_path = './skdh_test_data.npz'
    np.savez(save_path, time=time, accel=_data)

    data = skdh.io.ReadNumpyFile(allow_pickle=True).predict(file=save_path)
    return data


def main():
    
    # data = read_custom_txt()
    # print(type(data), data.keys())
    # print(data['accel'].shape)
    # plt.plot(data['accel'][:,0])
    # plt.show()
    # exit()

    data = read_custom_csv()
    # data = read_custom_npz()
    print(data)

    # print(f"Any nan in time: {np.any(np.isnan(data['time']))}")
    # print(f"Any nan in data: {np.any(np.isnan(data['accel']))}")
    # exit()
    accel_mag = np.linalg.norm(data['accel'], axis=1)
    print(accel_mag.shape)
    data_mean = skdh.features.Mean()
    data_entropy = skdh.features.SignalEntropy()
    print('data_mean', data_mean.compute(accel_mag))
    print('data_entropy', data_entropy.compute(accel_mag))


    # exit()
    print('<->'*25)
    print('PreprocessGaitBout')
    preprocess = skdh.gait.PreprocessGaitBout(
        correct_orientation=False,
        filter_cutoff=20.0,
        filter_order=4,
        )
    data_preprocess = preprocess.predict(
        time=data['time'], 
        accel=data['accel'],
        fs=148,
        )
    for k in data_preprocess.keys():
        print(f"[{k}] {data_preprocess[k]}")
        # print(f"Any NaN: {np.any(np.isnan(data_preprocess[k]))}")
    
    #################################################################################
    print('<->'*25)
    print('ApCwtGaitEvents')
    wave_events = skdh.gait.VerticalCwtGaitEvents()
    data_wave_events = wave_events.predict(
        time=data['time'], 
        accel=data['accel'], 
        accel_filt=data_preprocess['accel_filt'],
        v_axis=data_preprocess['v_axis'],
        v_axis_sign=data_preprocess['v_axis_sign'],
        mean_step_freq=data_preprocess['mean_step_freq'],
        fs = 148,
        )
    for k in data_wave_events.keys():
        print(f"[{k}] {data_wave_events[k]} - {len(data_wave_events[k])}")

    gait_events = skdh.gait.ApCwtGaitEvents()
    data_gait_event = gait_events.predict(
        time=data['time'], 
        accel=data['accel'], 
        accel_filt=data_preprocess['accel_filt'],
        ap_axis=data_preprocess['ap_axis'],
        ap_axis_sign=data_preprocess['ap_axis_sign'],
        mean_step_freq=data_preprocess['mean_step_freq'],
        fs= 148
        )
    for k in data_gait_event.keys():
        print(f"[{k}] {data_gait_event[k]} - {len(data_gait_event[k])}")

    #################################################################################
    print('<->'*25)
    print('CreateStridesAndQc')
    stride_events = skdh.gait.CreateStridesAndQc()
    data_stride_events = stride_events.predict(
        time=data['time'], 
        # initial_contacts=data_gait_event['initial_contacts'],
        # final_contacts=data_gait_event['final_contacts'],
        # mean_step_freq=data_preprocess['mean_step_freq'],
        initial_contacts=data_wave_events['initial_contacts'],
        final_contacts=data_wave_events['final_contacts'],
        mean_step_freq=data_preprocess['mean_step_freq'],
        fs= 148
    )
    for k in data_stride_events.keys():
        print(f"[{k}] {data_stride_events[k]} {np.mean(data_stride_events[k]):.4f}- {len(data_stride_events[k])}")
    

    #################################################################################
    print("#"*50)
    gait_lgbm = skdh.gait.PredictGaitLumbarLgbm()
    data_gait_lgbm = gait_lgbm.predict(
        time=data['time'], 
        accel=data['accel'], 
        # fs=148,
    )
    print(data_gait_lgbm)

    gait_lumbar = skdh.gait.GaitLumbar()
    data_gait_lumbar = gait_lumbar.predict(
        time=data['time'], 
        accel=data['accel'], 
        height=1.70,
        # fs=148,
    )
    for k in data_gait_lumbar.keys():
        print(f"[{k}] - {type(data_gait_lumbar[k])} {data_gait_lumbar[k][0]}")
        try:
            if np.any(np.isnan(data_gait_lumbar[k])):
                data_gait_lumbar[k][np.isnan(data_gait_lumbar[k])] = 0
            print(f"{np.mean(data_gait_lumbar[k]):.4f} - {data_gait_lumbar[k].shape}")
        except Exception as e:
            print(f'There was one error {e} so ...')
            pass

    # for k in data_stride_events.keys():
    #     print(f"[{k}] {data_stride_events[k]} - {len(data_stride_events[k])}")


    #################################################################################
    
    ts, ch = data['accel'].shape
    xf = np.linspace(1,ts, ts)
    plt.plot(
        xf, 
        data['accel'][:,0],
        c='blue',
        alpha=0.5)
    plt.plot(
        xf, 
        data['accel'][:,1],
        c='darkblue',
        alpha=0.5)
    plt.plot(
        xf, 
        data['accel'][:,2],
        c='skyblue',
        alpha=0.5)
    

    # plt.scatter(
    #     x=xf[data_gait_event['initial_contacts']],
    #     y=accel_mag[data_gait_event['initial_contacts']],
    #     marker='s',
    #     c='red',)
    # plt.scatter(
    #     x=xf[data_gait_event['final_contacts']],
    #     y=accel_mag[data_gait_event['final_contacts']],
    #     marker='*',
    #     c='purple',)
    plt.scatter(
        x=xf[data_stride_events['qc_initial_contacts']],
        y=accel_mag[data_stride_events['qc_initial_contacts']] * 0,
        marker='H',
        c='red',)
    plt.scatter(
        x=xf[data_stride_events['qc_final_contacts']],
        y=accel_mag[data_stride_events['qc_final_contacts']] * 0 + 1,
        marker='*',
        c='orange',)
    plt.scatter(
        x=xf[data_stride_events['qc_final_contacts_oppfoot']],
        y=accel_mag[data_stride_events['qc_final_contacts_oppfoot']] * 0 + 2,
        marker='h',
        c='sienna',)
    
    plt.scatter(
        x=xf[data_wave_events['initial_contacts']],
        y=accel_mag[data_wave_events['initial_contacts']] * 0 - 1,
        marker='^',
        c='teal',)
    plt.scatter(
        x=xf[data_wave_events['final_contacts']],
        y=accel_mag[data_wave_events['final_contacts']] * 0 - 1,
        marker='v',
        c='seagreen',)
    plt.show()
    exit()
    # gait_event_bout = skdh.gait.gait_metrics.GaitSpeedModel2()
    

    # gait_end = skdh.gait.gait_metrics.GaitEventEndpoint()
    # gait_sym = skdh.gait.gait_metrics.GaitSymmetryIndex()
    # swingtime = skdh.gait.gait_metrics.SwingTime()
    # print('swingtime: ', swingtime.predict())

    # g = GaitSymmetryIndex()
    # r = g.predict(data, fs=148)
    # print('GaitSymmetryIndex: ', r)
    print('*'*50)
    gait_lumbar = GaitLumbar(
        downsample=True,  # match original always downsampling
        height_factor=0.53,
        provide_leg_length=False,
        min_bout_time=8.0,
        max_bout_separation_time=0.5,
        gait_event_method='ap cwt',  # match original IC/FC estimation
        correct_orientation=True,
        filter_cutoff=20.0,
        filter_order=4,
        use_cwt_scale_relation=False,  # True or False
        wavelet_scale='default',  # default or float or int
        round_scale=True,  # originally rounded scale to integer, not required though
        max_stride_time=2.25,  # original QC threshold
        loading_factor=0.2,  # original QC factor for thresholds
        )
    
    out_gait_lumbar = gait_lumbar.predict(
        time=data['time'], 
        accel=data['accel'],
        fs=20,
        height=1.70,)
    for k in out_gait_lumbar.keys():
        print(f"[{k}] {out_gait_lumbar[k]}")
    
    # p = skdh.Pipeline()
    # p.add(GaitSymmetryIndex(), save_file="{file}_gait_results.csv")
    # p.add(g)
    # p.run(**data)

    

    # p.run(accel=data['accel'], time =data['time'])

    # data = skdh.io.ReadCwa().predict(file="example_data.cwa")
    # print(type(data), data.keys())

    # print(data['time'].shape)
    # print(data['time'][:10])
    # print(data['accel'].shape)
    # print(data['accel'][:10])

    
    # print(type(accel_mag), len(accel_mag))
    # fig = go.Figure(data=go.Scattergl(x=data['time'][::4] - data['time'][0], y=accel_mag[::4]))
    # fig.show()

if __name__ == '__main__':
    main()
    # app.run(debug=True)
