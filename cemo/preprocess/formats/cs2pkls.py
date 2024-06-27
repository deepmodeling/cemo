import numpy as np
import os
import pickle
import mrcfile
import numpy

'''
dtype([('uid', '<u8'), ('blob/path', 'S53'), ('blob/idx', '<u4'), ('blob/shape', '<u4', (2,)), ('blob/psize_A', '<f4'), ('blob/sign', '<f4'), ('blob/import_sig', '<u8'), ('ctf/type', 'S9'), ('ctf/exp_group_id', '<u4'), ('ctf/accel_kv', '<f4'), ('ctf/cs_mm', '<f4'), ('ctf/amp_contrast', '<f4'), ('ctf/df1_A', '<f4'), ('ctf/df2_A', '<f4'), ('ctf/df_angle_rad', '<f4'), ('ctf/phase_shift_rad', '<f4'), ('ctf/scale', '<f4'), ('ctf/scale_const', '<f4'), ('ctf/shift_A', '<f4', (2,)), ('ctf/tilt_A', '<f4', (2,)), ('ctf/trefoil_A', '<f4', (2,)), ('ctf/tetra_A', '<f4', (4,)), ('ctf/anisomag', '<f4', (4,)), ('ctf/bfactor', '<f4'), ('alignments3D/split', '<u4'), ('alignments3D/shift', '<f4', (2,)), ('alignments3D/pose', '<f4', (3,)), ('alignments3D/psize_A', '<f4'), ('alignments3D/error', '<f4'), ('alignments3D/error_min', '<f4'), ('alignments3D/resid_pow', '<f4'), ('alignments3D/slice_pow', '<f4'), ('alignments3D/image_pow', '<f4'), ('alignments3D/cross_cor', '<f4'), ('alignments3D/alpha', '<f4'), ('alignments3D/alpha_min', '<f4'), ('alignments3D/weight', '<f4'), ('alignments3D/pose_ess', '<f4'), ('alignments3D/shift_ess', '<f4'), ('alignments3D/class_posterior', '<f4'), ('alignments3D/class', '<u4'), ('alignments3D/class_ess', '<f4')])

(11221487418355032800, b'J12/imported/005213665113010562610_st_j425_d256.mrcs', 0, [256, 256], 1.05, -1., 18215134279155306700, b'imported', 23, 300., 2.7, 0.1, 11833.578, 11767.346, 4.037655, 0., 1., 0., [0., 0.], [0., 0.], [0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], 0., 0, [0.528125, 6.459375], [ 1.5672901, -2.1493263,  2.219451 ], 1.05, 2910.3135, 0., 0., 21.888172, 2938.776, 50.350586, 1., 1.150178, 0., 0., 0., 1., 0, 1.)

array([2.5600000e+02, 1.0500000e+00, 1.1833578e+04, 1.1767346e+04,
       2.3134058e+02, 3.0000000e+02, 2.7000000e+00, 1.0000000e-01,
       0.0000000e+00], dtype=float32)
'''

def cs2pkls(input_file_name:str,output_ctf_name:str,output_pose_name:str):
    data=np.load(input_file_name)

    data=np.load(input_file_name)
    data_len=data.shape[0]
    output_ctf=np.zeros(shape=(data_len,9),dtype=np.float32)
    output_ctf[:,0]=data['blob/shape'][:,0]
    output_ctf[:,1]=data['blob/psize_A']
    output_ctf[:,2]=data['ctf/df1_A']
    output_ctf[:,3]=data['ctf/df2_A']
    output_ctf[:,4]=data['ctf/df_angle_rad']*180/np.pi # convert rad to degree
    output_ctf[:,5]=data['ctf/accel_kv']
    output_ctf[:,6]=data['ctf/cs_mm']
    output_ctf[:,7]=data['ctf/amp_contrast']
    output_ctf[:,8]=data['ctf/phase_shift_rad']*180/np.pi
    pickle.dump(output_ctf,open(output_ctf_name,'wb'))

    output_pose_rot=data['alignments3D/pose']  # src shape: (data_len,3)
    # target shape: np.zeros((data_len,3,3),dtype=np.float32)
    theta=np.linalg.norm(output_pose_rot,ord=2,axis=-1,keepdims=True)
    v=output_pose_rot/theta # v: (datalen,3)
    v=np.tile(np.expand_dims(v,-1),(1,1,3)) # (datalen,3,3)
    rx=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    ry=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    rz=np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    r=np.array(map(lambda v:rx*v[0]+ry*v[1]+rz*v[2],v))
    rot_mat=r.transpose(0,2,1)
    shift_mat=data['alignments3D/shift']/np.expand_dims(data['blob/shape'][:,0],-1)
    pickle.dump((rot_mat,shift_mat),open(output_pose_name,'wb'))
