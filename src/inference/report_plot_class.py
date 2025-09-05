import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from scipy.interpolate import interp1d
import math
import warnings
import concurrent.futures
import multiprocessing as mp

warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class ECGVisualizer:
    def __init__(self, path_base="./utils/", use_mp=False):
        self.use_mp = use_mp
        self.path_base = path_base
        self.path_template = path_base + "template/"
        self.path_image = path_base + ""

        self.template_hr = pd.read_csv(self.path_template + "_table_hazard_ratio.csv", encoding="cp1252")
        self.template_sp = pd.read_csv(self.path_template + "_table_scatter_plot.csv")
        self.template_sp["age"] = pd.to_numeric(self.template_sp["age"], errors='coerce')
        self.template_sp = self.template_sp.dropna().reset_index(drop=True)
        self.template_sp["pred_cia"] = np.where(self.template_sp["pred_cia"] < 0.075, self.template_sp["pred_cia"] / 2, self.template_sp["pred_cia"])
        self.template_sp["pred_cia"] = np.where(self.template_sp["pred_cia"] > 0.925, (self.template_sp["pred_cia"] + 1) / 2, self.template_sp["pred_cia"])

        contour_bar = np.load(self.path_template + "_table_bar_200.npz")
        contour_dot = np.load(self.path_template + "_table_dot_200.npz")
        self.cbar_x = contour_bar['xi']
        self.cbar_y_upper = contour_bar['yi_upper']
        self.cbar_y_lower = contour_bar['yi_lower']
        self.cdot_x = contour_dot['xi']
        self.cdot_y_upper = contour_dot['yi_upper']
        self.cdot_y_lower = contour_dot['yi_lower']
        self.macai_button = plt.imread(self.path_template + '_macai_button.png')

        self.dict_cdot = dict()
        self.n_dpi = 8
        self.list_dpi = (np.linspace(0, 200, num=self.n_dpi + 1) / 2).astype('int')[1:] * 2

        for dpi in self.list_dpi:
            contour_dot = np.load(self.path_template + f"_table_dot_{dpi}.npz")
            cdot_x = contour_dot['xi']
            cdot_y_upper = contour_dot['yi_upper']
            cdot_y_lower = contour_dot['yi_lower']
            self.dict_cdot[dpi] = dict()
            self.dict_cdot[dpi]['q1'] = cdot_y_upper[cdot_x >= 0]
            self.dict_cdot[dpi]['q3'] = cdot_y_lower[cdot_x <= 0]
            self.dict_cdot[dpi]['q4'] = cdot_y_lower[cdot_x >= 0]
            self.dict_cdot[dpi]['ref'] = np.stack([cdot_x[cdot_x >= 0], cdot_x[cdot_x <= 0]])

        self.color_afib = (248, 57, 90)
        self.color_cia = (57, 90, 248)
        self.att_modifier = 100

        if self.use_mp:
            mp.set_start_method('spawn', force=True)

    def run_resampling(self, arrays, fs_original, fs_target):
        gcd = np.gcd(fs_original, fs_target)
        up = fs_target // gcd
        down = fs_original // gcd

        resampled_arrays = resample_poly(np.swapaxes(arrays, 0, 1), up, down)
        resampled_arrays = np.swapaxes(resampled_arrays, 0, 1)
        return resampled_arrays

    def create_segments_vertical(self, xc, yc_lower, yc_upper):
        points_lower = np.concatenate([xc.reshape(-1,1), yc_lower.reshape(-1,1)], axis=1).reshape(-1,1,2)
        points_upper = np.concatenate([xc.reshape(-1,1), yc_upper.reshape(-1,1)], axis=1).reshape(-1,1,2)
        segments = np.concatenate([points_lower, points_upper], axis=1)
        return segments

    def create_segments_horizontal(self, xc_lower, xc_upper, yc):
        points_lower = np.concatenate([xc_lower.reshape(-1,1), yc.reshape(-1,1)], axis=1).reshape(-1,1,2)
        points_upper = np.concatenate([xc_upper.reshape(-1,1), yc.reshape(-1,1)], axis=1).reshape(-1,1,2)
        segments = np.concatenate([points_lower, points_upper], axis=1)
        return segments

    def color_set_alpha_one(self, color, alpha=1):
        color_scaled = tuple([x/255 for x in color])

        if alpha != 1:
            color_scaled = tuple([(1-alpha+alpha*x) for x in color_scaled])
        
        return color_scaled

    def colored_line_segments(self, ax, segments, colors, cmap, val_max=None, val_min=None, linewidth=1, zorder=2):
        if val_max is None:
            val_max = colors.max()
            val_min = colors.min()

        norm = Normalize(vmin=val_min, vmax=val_max)
        lc = LineCollection(segments, cmap=cmap, antialiased=True, zorder=zorder, norm=norm)
        lc.set_array(colors)
        lc.set_linewidth(linewidth)
        return ax.add_collection(lc)

    def colored_gradient(self, ax, fig, x, y, colors,
                          linewidth = 1,
                          alpha_outer=0.1, alpha_inner=0.5,
                          col_low=(0.1,0.1,0.1), col_high=(0.8,0.1,0.1),
                          num_layers=5,
                          val_max=None, val_min=None,
                          min_pixel=2, max_pixel=6, angle_shadowing=False, zorder=2):

        dict_contour = self.get_gradient_contour_norm(x, y,
                                            w_pixel_min=min_pixel, w_pixel_max=max_pixel, w_pixel_n=num_layers,
                                            angle_shadowing=angle_shadowing)

        list_alpha = np.linspace(alpha_inner, alpha_outer, num_layers)

        for i, t_alpha in enumerate(list_alpha):
            xc, yc_lower, yc_upper = dict_contour[i]['xc'], dict_contour[i]['yc_lower'], dict_contour[i]['yc_upper']

            n_pad = xc.shape[0] - x.shape[0]
            if n_pad == 0:
                colors_pad = colors
            elif n_pad < 0:
                raise "unexpected colored gradient"
            else:
                pre_pad = n_pad//2
                post_pad = n_pad - pre_pad
                first_vals = np.full(pre_pad, colors[0])
                last_vals = np.full(post_pad, colors[-1])
                colors_pad = np.concatenate([first_vals, colors, last_vals])

            segments = self.create_segments_vertical(xc, yc_lower, yc_upper)
            cmap = LinearSegmentedColormap.from_list('custom', [col_low+(t_alpha,), col_high+(t_alpha,)])
            line = self.colored_line_segments(ax, segments, colors_pad, cmap, linewidth=linewidth, val_max=val_max, val_min=val_min, zorder=zorder)
            cbar = fig.colorbar(line)
            cbar.remove()

    def interpolate(self, x, y, x_n, kind='linear'):
        interpolation_function = interp1d(x, y, kind=kind, fill_value="extrapolate")
        y_n = interpolation_function(x_n)
        return y_n

    def get_gradient_contour_norm(self, x, y,
                            w_pixel_min=2, w_pixel_max=6, w_pixel_n = 5,
                            angle_shadowing=False):        
        x_max = x.max(); x_min = x.min()
        x_range = x_max - x_min
        dpi = x.shape[0]

        x_norm = (x-x_min)/x_range * dpi
        y_norm = (y-x_min)/x_range * dpi
        y_norm_min = y_norm.min()
        y_norm -= y_norm_min

        # plt.plot(x_norm, y_norm)
        # plt.gca().set_aspect(1)
        # plt.show()

        x_int = np.arange(dpi)
        y_int = self.interpolate(x_norm, y_norm, x_int)
        y_int = np.round(y_int).astype('int')

        # hard coding here, only for plot_margin_shadowing
        if angle_shadowing:
            x_int = np.concatenate([x_int, [x_int[-1]]], axis=0)
            y_int[-1] = y_int[-2]
            y_int = np.concatenate([y_int, [y_norm.max()]], axis=0)
            y_int = np.round(y_int).astype('int')

        # moving (it create clean margin)
        x_int_range = dpi
        y_int_range = y_int.max()-y_int.min()
        margin_pre = int(min([x_int_range, y_int_range])/2)
        margin_post = min([x_int_range, y_int_range]) - margin_pre

        x_int += margin_pre
        y_int += margin_pre

        # plt.scatter(x_int, y_int)
        # plt.gca().set_aspect(1)
        # plt.show()

        t_image = np.zeros((x_int_range+margin_pre+margin_post, y_int_range+margin_pre+margin_post))
        t_image[x_int, y_int] = 1

        if angle_shadowing:
            t_image[x_int.max(), y_int.min():y_int.max()] = 1

        # plt.imshow(t_image.swapaxes(0,1), cmap='gray', origin='lower')
        # plt.show()

        t_image = torch.tensor(t_image).unsqueeze(0).unsqueeze(0).cuda()

        list_conv = []
        list_w_pixel=np.linspace(w_pixel_min, w_pixel_max, num=w_pixel_n)
        for w_pixel in list_w_pixel:
            t_conv = np.zeros((w_pixel_max*2+1, w_pixel_max*2+1))
            center = (w_pixel_max-0.5, w_pixel_max-0.5); radius = w_pixel
            wy, wx = np.ogrid[:w_pixel_max*2+1, :w_pixel_max*2+1]
            distance_from_center = np.sqrt((wx - center[0])**2 + (wy - center[1])**2)
            t_conv[distance_from_center <= radius] = 1
            list_conv.append(t_conv)

        list_conv = np.stack(list_conv, axis=0)
        list_conv = torch.tensor(list_conv).unsqueeze(0).swapaxes(0,1).cuda()
        output = F.conv2d(t_image, list_conv, padding=(w_pixel_max, w_pixel_max))
        output = output.squeeze(0).cpu().detach().numpy()

        del t_image, list_conv
        torch.cuda.empty_cache()

        output = np.where(output>0, 1, 0)

        if angle_shadowing:
            # output[:, x_int, y_int] = 0
            # output[:, x_int.max(), y_int.min():y_int.max()] = 0
            output[:, :x_int.max(), y_int.min():] = 0

        # plt.imshow(output[-1,:,:].swapaxes(0,1), cmap='gray', origin='lower')
        # plt.show()

        dict_contour = dict()
        for idx, arr in enumerate(output):
            cumsum_fw = np.cumsum(arr, axis=1)
            cumsum_bw = np.cumsum(arr[:, ::-1], axis=1)[:, ::-1]

            # Indices of the first 1
            lower_indices = (cumsum_fw == 1) & (arr == 1)
            lower_idx = np.argmax(lower_indices, axis=1)
            lower_idx[arr.sum(axis=1) == 0] = -1  # or another value indicating no 1's

            # Indices of the last 1
            upper_indices = (cumsum_bw == 1) & (arr == 1)
            upper_idx = np.argmax(upper_indices, axis=1)
            upper_idx[arr.sum(axis=1) == 0] = -1  # or another value indicating no 1's

            xc_lower = np.where(lower_idx!=-1)[0]
            xc_upper = np.where(upper_idx!=-1)[0]
            assert np.array_equal(xc_lower, xc_upper)

            xc = xc_lower
            yc_lower = lower_idx[xc]
            yc_upper = upper_idx[xc]

            # rescaling for the original value of x and y
            xc = (xc - margin_pre) / dpi * x_range + x_min
            yc_lower = (yc_lower - margin_pre + y_norm_min) / dpi * x_range + x_min
            yc_upper = (yc_upper - margin_pre + y_norm_min) / dpi * x_range + x_min

            dict_contour[idx] = dict()
            dict_contour[idx]['xc'] = xc
            dict_contour[idx]['yc_lower'] = yc_lower
            dict_contour[idx]['yc_upper'] = yc_upper

        return dict_contour

    def get_gradient_contour_norm(self, x, y,
                            w_pixel_min=2, w_pixel_max=6, w_pixel_n = 5,
                            angle_shadowing=False):
        x_max = x.max(); x_min = x.min()
        x_range = x_max - x_min
        dpi = x.shape[0]

        x_norm = (x-x_min)/x_range * dpi
        y_norm = (y-x_min)/x_range * dpi
        y_norm_min = y_norm.min()
        y_norm -= y_norm_min
        x_int = np.arange(dpi)
        y_int = self.interpolate(x_norm, y_norm, x_int)
        y_int = np.round(y_int).astype('int')

        # hard coding here, only for plot_margin_shadowing
        if angle_shadowing:
            x_int = np.concatenate([x_int, [x_int[-1]]], axis=0)
            y_int[-1] = y_int[-2]
            y_int = np.concatenate([y_int, [y_norm.max()]], axis=0)
            y_int = np.round(y_int).astype('int')

        # moving (it create clean margin)
        x_int_range = dpi
        y_int_range = y_int.max()-y_int.min()
        margin_pre = int(min([x_int_range, y_int_range])/2)
        margin_post = min([x_int_range, y_int_range]) - margin_pre

        x_int += margin_pre
        y_int += margin_pre

        t_image = np.zeros((x_int_range+margin_pre+margin_post, y_int_range+margin_pre+margin_post))
        t_image[x_int, y_int] = 1

        if angle_shadowing:
            t_image[x_int.max(), y_int.min():y_int.max()] = 1

        t_image = torch.tensor(t_image).unsqueeze(0).unsqueeze(0).cuda()

        list_conv = []
        list_w_pixel=np.linspace(w_pixel_min, w_pixel_max, num=w_pixel_n)
        for w_pixel in list_w_pixel:
            t_conv = np.zeros((w_pixel_max*2+1, w_pixel_max*2+1))
            center = (w_pixel_max-0.5, w_pixel_max-0.5); radius = w_pixel
            wy, wx = np.ogrid[:w_pixel_max*2+1, :w_pixel_max*2+1]
            distance_from_center = np.sqrt((wx - center[0])**2 + (wy - center[1])**2)
            t_conv[distance_from_center <= radius] = 1
            list_conv.append(t_conv)

        list_conv = np.stack(list_conv, axis=0)
        list_conv = torch.tensor(list_conv).unsqueeze(0).swapaxes(0,1).cuda()
        output = F.conv2d(t_image, list_conv, padding=(w_pixel_max, w_pixel_max))
        output = output.squeeze(0).cpu().detach().numpy()

        del t_image, list_conv
        torch.cuda.empty_cache()

        output = np.where(output>0, 1, 0)

        if angle_shadowing:
            output[:, :x_int.max(), y_int.min():] = 0

        dict_contour = dict()
        for idx, arr in enumerate(output):
            cumsum_fw = np.cumsum(arr, axis=1)
            cumsum_bw = np.cumsum(arr[:, ::-1], axis=1)[:, ::-1]

            # Indices of the first 1
            lower_indices = (cumsum_fw == 1) & (arr == 1)
            lower_idx = np.argmax(lower_indices, axis=1)
            lower_idx[arr.sum(axis=1) == 0] = -1  # or another value indicating no 1's

            # Indices of the last 1
            upper_indices = (cumsum_bw == 1) & (arr == 1)
            upper_idx = np.argmax(upper_indices, axis=1)
            upper_idx[arr.sum(axis=1) == 0] = -1  # or another value indicating no 1's

            xc_lower = np.where(lower_idx!=-1)[0]
            xc_upper = np.where(upper_idx!=-1)[0]
            assert np.array_equal(xc_lower, xc_upper)

            xc = xc_lower
            yc_lower = lower_idx[xc]
            yc_upper = upper_idx[xc]

            # rescaling for the original value of x and y
            xc = (xc - margin_pre) / dpi * x_range + x_min
            yc_lower = (yc_lower - margin_pre + y_norm_min) / dpi * x_range + x_min
            yc_upper = (yc_upper - margin_pre + y_norm_min) / dpi * x_range + x_min

            dict_contour[idx] = dict()
            dict_contour[idx]['xc'] = xc
            dict_contour[idx]['yc_lower'] = yc_lower
            dict_contour[idx]['yc_upper'] = yc_upper

        return dict_contour

    def draw_rectangles(self, x1, x2, y1, y2, color="white", zorder=2):
        rectangle = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=0, edgecolor='none', facecolor=color, zorder=zorder)
        # zorder default ... patch = 1, line = 2, text = 3
        return rectangle

    def round_border(self, ax, lw=2, color='black', rad=0.1):
        fig = ax.figure
        bbox = ax.get_position()
        bbox = FancyBboxPatch((bbox.x0, bbox.y0), bbox.width, bbox.height,
                            boxstyle=f"round,pad=0.1,rounding_size={rad}",
                            edgecolor=color, facecolor='none', lw=lw,
                            transform=fig.transFigure, zorder=3)
        fig.patches.append(bbox)

    # plot_circular_attention
    def norm_encoder(self, x, tmin=0.2, tmax=1.0):
        xmin = np.min(x)
        xmax = np.max(x)
        norm_x = (x - xmin) / (xmax - xmin) if xmax != xmin else np.zeros_like(x)
        scaled_x = norm_x * (tmax - tmin) + tmin
        return scaled_x

    def norm_decoder(self, scaled_x, xmin, xmax, tmin=0.2, tmax=1.0):
        norm_x = (scaled_x - tmin) / (tmax - tmin) if tmax != tmin else np.zeros_like(scaled_x)
        x = norm_x * (xmax - xmin) + xmin
        return x

    def adaptive_format(self, x, data_range):
        if data_range == 0:
            return f"{x:.1f}"  # Default to 1 decimal place if range is 0
        significant_figures = int(np.floor(-np.log10(data_range))) + 2  # Adding 2 for extra precision
        return f"{x:.{significant_figures}f}"

    # plot_scatter_plot
    def discretizer(self, array, n_bin):
        min_val = np.min(array)
        max_val = np.max(array)
        bins = np.linspace(min_val, max_val, n_bin+1)
        discrete_array = np.digitize(array, bins, right=False) - 1
        discrete_array[discrete_array == n_bin] = n_bin - 1
        range_min = bins[:-1]  # Exclude the last edge which is the maximum value
        range_max = bins[1:]   # Exclude the first edge which is the minimum value

        return discrete_array, range_min, range_max

    def count_value_pairs(self, disc_age, disc_scr, max_value=19):
        result, _, _ = np.histogram2d(disc_age, disc_scr, bins=(max_value+1, max_value+1), range=[[0, max_value], [0, max_value]])
        # result = np.swapaxes(result, 0, 1)
        result = np.log1p(result)
        result = result/result.max()
        v_min = result[result!=0].min()
        result[result==0] = v_min/2
        return result

    def create_color_gradient(self, n_seg, color_list, threshold_list):
        assert (threshold_list[0] == 0) & threshold_list[-1] == 1, "invalid range"
        assert len(color_list) == len(threshold_list), "invalid length"

        color_list = [np.array(color) for color in color_list]
        color_array = np.zeros((n_seg, 3))

        # Loop over each segment
        for t in range(len(threshold_list) - 1):
            start_idx = int(n_seg * threshold_list[t])
            end_idx = int(n_seg * threshold_list[t + 1])
            for i in range(start_idx, end_idx):
                ratio = (i - start_idx) / (end_idx - start_idx)
                color_array[i] = (1 - ratio) * color_list[t] + ratio * color_list[t + 1]
        
        return color_array

    def color_set_alpha_one(self, color, alpha=1):
        color_scaled = tuple([x/255 for x in color])

        if alpha != 1:
            color_scaled = tuple([(1-alpha+alpha*x) for x in color_scaled])
        
        return color_scaled

    def calculate_most_important_lead_idx(self, output_dict, arrhythmia="afib"):
        """Calculate the most important lead index based on attention weights"""
        npy_attention = output_dict[f'npy_attention_{arrhythmia}']
        lead_importance = np.mean(np.abs(npy_attention) ** 2, axis=1)
        most_important_lead_idx = np.argmax(lead_importance)
        return most_important_lead_idx
    
    def get_lead_name(self, lead_idx):
        """Convert lead index to lead name"""
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        return lead_names[lead_idx]
    
    def calculate_most_important_lead_info(self, output_dict, arrhythmia="afib"):
        """Calculate the most important lead index and name based on attention weights"""
        lead_idx = self.calculate_most_important_lead_idx(output_dict, arrhythmia)
        lead_name = self.get_lead_name(lead_idx)
        return lead_idx, lead_name

    def _plot_ecg_attention_one_lead(self, output_dict, arrhythmia = "afib", language="eng", save_directory=None):
        color_heat = self.color_afib if arrhythmia == "afib" else self.color_cia
        # ECG 데이터 로드
        npy_ecg_raw         = output_dict['ecg_orig']
        npy_ecg_filtered    = output_dict['ecg_filtered']
        npy_attention       = output_dict[f'npy_attention_{arrhythmia}']

        # attention weight가 가장 높은 리드 찾기: 12개 리드에 대해 각각 절대값 추출을 위해 제곱 후 평균 및 보정
        # lead_importance             = np.sqrt(np.mean(npy_attention**2, axis=1))  # (12, )
        lead_importance             = np.mean(np.abs(npy_attention) ** 2, axis=1)   # (12, )
        most_important_lead_idx     = np.argmax(lead_importance)
        npy_ecg_raw                 = npy_ecg_raw[most_important_lead_idx]          # (2500, )
        npy_ecg_filtered            = npy_ecg_filtered[most_important_lead_idx]     # (2500, )
        npy_attention               = npy_attention[most_important_lead_idx]        # (2500, )

        # 초당 데이터 수 (250Hz로 설정)
        sampling_rate       = 256
        elongation_factor   = 1
        color_bright        = (254, 196, 20)
        color_bright        = self.color_set_alpha_one(color_bright)
        color_black         = (0.15, 0.15, 0.15)
        background_color    = self.color_set_alpha_one(color_heat, alpha=0.05)
        color_heat          = self.color_set_alpha_one(color_heat)

        # Resampling, if needed
        if elongation_factor != 1:
            npy_ecg_filtered    = self.run_resampling(npy_ecg_filtered, sampling_rate, sampling_rate*elongation_factor)
            npy_attention       = self.run_resampling(npy_attention, sampling_rate, sampling_rate*elongation_factor)

        time_axis = np.arange(npy_ecg_filtered.shape[0]) / (sampling_rate*elongation_factor)

        xtick = 0.2 # 0.2 sec ... standard paper speed = 25mm/sec -> 1 small square (1mm) = 0.04 sec (not shown), 5 small square = 1 large square = 0.2 sec
        ytick = 0.5 # 0.5 mV
        # aspect_ratio = xtick/ytick

        current_amplitude = np.max(np.abs(npy_ecg_filtered))
        target_amplitude = 2.0
        amplitude_scale = target_amplitude / (current_amplitude + 1e-8)

        aspect_ratio = (xtick/ytick) * amplitude_scale
        aspect_ratio = np.clip(aspect_ratio, 0.1, 2.0)  # 최소 0.1, 최대 2.0으로 제한

        xmin    = 0 - (xtick*5)
        xmax    = 10 + (xtick*5) # 총 10초 + a
        ymin    = -xtick*3
        ymax    = (xtick*1) + (xtick*3) # 각 lead 중심선 사이 거리 1칸

        xmin_ext = xmin-xtick*3
        xmax_ext = xmax+xtick*3
        ymin_ext = ymin-xtick*3
        ymax_ext = ymax+xtick*3

        # Figure start
        fig, ax = plt.subplots(figsize=(15, 3))
        ax.set_xlim(xmin_ext, xmax_ext)
        ax.set_ylim(ymin_ext, ymax_ext)

        # Update y-axis limits for center alignment
        ymin = -xtick * 4  # 더 넉넉하게 아래로 확장
        ymax = xtick * 5   # 더 넉넉하게 위로 확장

        # x,y 축 범위 설정
        ax.set_xlim(0, 10)
        ax.set_ylim(ymin, ymax)
        ax.set_facecolor(background_color)

        # 테두리 설정
        ax.spines['top'].set_linewidth(3)           # 위쪽 테두리 굵게
        ax.spines['bottom'].set_linewidth(3)        # 아래쪽 테두리 굵게
        ax.spines['top'].set_color(color_heat)      # 위쪽 테두리 색상
        ax.spines['bottom'].set_color(color_heat)   # 아래쪽 테두리 색상
        ax.spines['left'].set_visible(False)        # 좌측 테두리 제거
        ax.spines['right'].set_visible(False)       # 우측 테두리 제거

        ax.set_xticklabels([])                      # x축 눈금 표시 제거
        ax.set_yticklabels([])                      # y축 눈금 표시 제거

        ax.xaxis.set_major_locator(plt.MultipleLocator(xtick))
        ax.yaxis.set_major_locator(plt.MultipleLocator(xtick))

        ax.tick_params(axis='both', which='major', width=0.5, length=0)
        ax.grid(True, color=color_heat, alpha=0.2)
        ax.set_aspect(1)

        x = time_axis.copy()
        y = npy_ecg_filtered * aspect_ratio
        x_raw = x + xtick/2.5
        y_raw = npy_ecg_raw * aspect_ratio - xtick/2.5

        # raw ECG
        ax.plot(x_raw, y_raw, color=(0.5, 0.5, 0.5), alpha=0.5, linewidth=2)

        # filtered ECGs
        points = np.array([x, y]).T.reshape(-1, 1, 2) # (N_point, 1, (x,y))
        segments = np.concatenate([points[:-1], points[1:]], axis=1) # (N_seg, (x1,y1), (x2,y2))
        cmap = LinearSegmentedColormap.from_list('custom', [color_black + (1,), color_heat + (1,)])
        line = self.colored_line_segments(ax, segments, npy_attention, cmap=cmap, linewidth=3)
        cbar = fig.colorbar(line, aspect=40, pad=0.01, shrink=0.5)
        cbar.remove()

        plt.subplots_adjust(left=-0.02, right=1.0, bottom=0.02, top=1, hspace=0, wspace=0)
        plt.savefig(f'{save_directory}/plot_ecg_attention_one_lead_{language}_{arrhythmia}.png', dpi=300)
        fig.clf()
        plt.clf()
        plt.close()
        
        return most_important_lead_idx

    def _plot_ecg_attention(self, output_dict, arrhythmia = "afib", language="eng", save_directory=None):
        color_heat = self.color_afib if arrhythmia == "afib" else self.color_cia

        npy_ecg_raw = output_dict['ecg_orig'].copy()
        npy_ecg_filtered = output_dict['ecg_filtered'].copy()
        npy_attention = np.abs(output_dict[f'npy_attention_{arrhythmia}'].copy() * self.att_modifier)

        # Hard coding
        sampling_rate = 256
        elongation_factor = 1
        list_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        xtick = 0.2
        ytick = 0.5
        
        # 리드별 최대 허용 진폭 계산 (격자 범위 내에서)
        max_allowed_amplitude_per_lead = ytick * 2.5  # 각 리드당 2.5 격자 범위
        ypad = xtick * 7  # 리드 간 간격
        
        # 안전한 간격 확보를 위해 ypad 증가
        safe_margin = 0.05  # 30% 안전 여백
        ypad = ypad * (1 + safe_margin)
        
        # 리드별 개별 스케일링 계산
        lead_scales = []
        for idx in range(12):
            lead_amplitude = np.max(np.abs(npy_ecg_filtered[idx, :]))
            # 각 리드가 허용 범위를 넘지 않도록 스케일 조정
            if lead_amplitude > 0:
                scale = max_allowed_amplitude_per_lead / lead_amplitude
                # 너무 작은 신호는 최소 크기 보장
                scale = max(scale, 0.1)
                # 너무 큰 신호는 제한
                scale = min(scale, 2.0)
            else:
                scale = 1.0
            lead_scales.append(scale)
        
        # 전체적인 aspect_ratio는 기본값 사용
        base_aspect_ratio = xtick / ytick
        
        val_max = npy_attention.max()
        val_min = npy_attention.min()

        xmin = 0; xmax = 10+xtick*8
        ymin = -xtick*3; ymax = (xtick*7)*11 + xtick*3
        
        ymax = ypad * 11 + xtick*3

        color_black = (0.15, 0.15, 0.15)
        ba = 0.05
        background_color = self.color_set_alpha_one(color_heat, alpha=ba)
        color_heat = self.color_set_alpha_one(color_heat)

        if elongation_factor != 1:
            npy_ecg_filtered = self.run_resampling(npy_ecg_filtered, sampling_rate, sampling_rate*elongation_factor)
            npy_attention = self.run_resampling(npy_attention, sampling_rate, sampling_rate*elongation_factor)
        time_axis = np.arange(npy_ecg_filtered.shape[1]) / (sampling_rate*elongation_factor)

        def draw_rectangles(x1, x2, y1, y2, color="white", zorder=2):
            rectangle = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=0, edgecolor='none', facecolor=color, zorder=zorder)
            return rectangle

        xmin_ext = xmin-xtick*3; xmax_ext = xmax+xtick*3
        ymin_ext = ymin-xtick*3; ymax_ext = ymax+xtick*3
        mx = [xmin_ext, xmin-xtick, xmin, xmax, xmax+xtick, xmax_ext]
        my = [ymin_ext, ymin-xtick, ymin, ymax, ymax+xtick, ymax_ext]

        # Figure 크기 조정 (높이 증가)
        fig_height = 7 * (1 + safe_margin)  # 안전 여백만큼 높이 증가
        fig, ax = plt.subplots(figsize=(6, fig_height))

        ax.set_xlim(xmin_ext, xmax_ext)
        ax.set_ylim(ymin_ext, ymax_ext) 
        ax.set_facecolor(background_color)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(plt.MultipleLocator(xtick))
        ax.yaxis.set_major_locator(plt.MultipleLocator(xtick))
        ax.tick_params(axis='both', which='major', width=0.5, length=0)
        ax.grid(True, color=color_heat, alpha=0.2, zorder=1)
        ax.set_aspect(1)

        ax.add_patch(draw_rectangles(mx[0], mx[5], my[0], my[2]))
        ax.add_patch(draw_rectangles(mx[0], mx[5], my[3], my[5]))
        ax.add_patch(draw_rectangles(mx[0], mx[2], my[0], my[5]))
        ax.add_patch(draw_rectangles(mx[3], mx[5], my[0], my[5]))

        ax.add_patch(draw_rectangles(mx[0], mx[5], my[0], my[1]))
        ax.add_patch(draw_rectangles(mx[0], mx[5], my[4], my[5]))
        ax.add_patch(draw_rectangles(mx[0], mx[1], my[0], my[5]))
        ax.add_patch(draw_rectangles(mx[4], mx[5], my[0], my[5]))

        paper_margin_x = np.array([mx[4], mx[4], mx[1], mx[1], mx[4]])
        paper_margin_y = np.array([my[1], my[4], my[4], my[1], my[1]])
        ax.plot(paper_margin_x, paper_margin_y, color=color_black+(0.05,), linewidth=0.75, zorder=4)

        plot_margin_x = np.array([mx[3], mx[3], mx[2], mx[2], mx[3]])
        plot_margin_y = np.array([my[2], my[3], my[3], my[2], my[2]])
        ax.plot(plot_margin_x, plot_margin_y, color=color_heat+(0.5,), linewidth=0.75, zorder=4)

        # Drawing layers with individual scaling
        list_x = []
        list_y = []
        list_att = []

        for idx in range(12):
            t_y = 11-idx

            s = xtick*0.2
            move_x = xtick*6; move_tx = xtick*2
            move_y = t_y*ypad

            # R-wave 마커
            rpulse_x = np.array([move_x-xtick-s, move_x-xtick, move_x-xtick, move_x, move_x, move_x+s])
            rpulse_y = np.array([move_y, move_y, move_y+xtick*2, move_y+xtick*2, move_y, move_y])
            ax.plot(rpulse_x-s*2.5, rpulse_y, color=color_black, linewidth=0.75)

            # 리드 레이블
            ax.text(move_tx, move_y, list_leads[idx], color=color_black, weight="bold",
                    fontsize=8,
                    horizontalalignment='center', verticalalignment='center')

            x = time_axis.copy() + move_x
            
            # 개별 리드 스케일링 적용
            individual_scale = base_aspect_ratio * lead_scales[idx]
            y = npy_ecg_filtered[idx,:] * individual_scale + move_y
            x_raw = x + xtick/2.5
            y_raw = npy_ecg_raw[idx,:] * individual_scale + move_y - xtick/2.5
            att = npy_attention[idx,:]

            # 신호 클리핑 (격자 범위 내로 제한)
            y_min_limit = move_y - max_allowed_amplitude_per_lead
            y_max_limit = move_y + max_allowed_amplitude_per_lead
            y = np.clip(y, y_min_limit, y_max_limit)
            y_raw = np.clip(y_raw, y_min_limit - xtick/2.5, y_max_limit - xtick/2.5)

            kernel = np.ones(25)/25
            att_conv = np.convolve(att, kernel, mode='same')

            # raw ECG
            ax.plot(x_raw, y_raw, color=(0.5, 0.5, 0.5), alpha=0.3, linewidth=0.8)

            # filtered ECGs
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            att_avg = (att[:-1] + att[1:]) / 2
            cmap = LinearSegmentedColormap.from_list('custom', [color_black + (1,), color_heat + (1,)])
            line = self.colored_line_segments(ax, segments, att, cmap=cmap, linewidth=0.8, val_max=val_max, val_min=val_min, zorder=2)
            cbar = fig.colorbar(line, aspect=40, pad=0.01, shrink=0.5)
            cbar.remove()

            list_x.append(x)
            list_y.append(y)
            list_att.append(att)

        list_x = np.concatenate(list_x, axis=0)
        list_y = np.concatenate(list_y, axis=0)
        list_att = np.concatenate(list_att, axis=0)

        shadow_length = xtick*1.5
        list_s = np.linspace(0, shadow_length, num = self.list_dpi.shape[0]+1)
        list_s = list_s[1:]
        list_a = np.linspace(0.004, 0.004, num = self.list_dpi.shape[0])

        plot_x = np.array([mx[1],mx[4],mx[4]])
        plot_y = np.array([my[1],my[1],my[4]])

        for (dpi, ts, ta) in zip(self.list_dpi, list_s, list_a):
            set_ref0 = self.dict_cdot[dpi]['ref'][0,:]*2*ts
            set_ref1 = self.dict_cdot[dpi]['ref'][1,:]*2*ts
            set_q1 = self.dict_cdot[dpi]['q1']*2*ts
            set_q3 = self.dict_cdot[dpi]['q3']*2*ts
            set_q4 = self.dict_cdot[dpi]['q4']*2*ts

            start_x = set_ref0
            start_y1 = set_ref1[::-1]
            start_y2 = set_q4
            start_y = np.where(start_y1>start_y2, start_y1, start_y2)

            start_xn = -start_y
            start_yn = -start_x

            set1_start = np.stack([start_x+plot_x[1], start_y+plot_y[1]], axis=1).reshape(-1, 1, 2)
            set1_end = np.stack([set_ref0+plot_x[2], set_q1+plot_y[2]-shadow_length], axis=1).reshape(-1, 1, 2)
            set1 = np.concatenate([set1_start, set1_end], axis=1)

            set2_start = np.stack([start_xn+plot_x[1], start_yn+plot_y[1]], axis=1).reshape(-1, 1, 2)
            set2_end = np.stack([set_q3[::-1]+plot_x[0]+shadow_length, set_ref1[::-1]+plot_y[0]], axis=1).reshape(-1, 1, 2)
            set2 = np.concatenate([set2_start, set2_end], axis=1)

            segments = np.concatenate([set1, set2], axis=0)
            color_null = np.repeat(0, segments.shape[0])
            cmap = LinearSegmentedColormap.from_list('custom', [color_black+(ta,), color_black+(ta,)])
            line = self.colored_line_segments(ax, segments, color_null, cmap, linewidth=0.5, zorder=2.5)
            cbar = fig.colorbar(line)
            cbar.remove()

        # Legends only (Attention weight)
        cmap = LinearSegmentedColormap.from_list('custom', [color_black+(1,), color_heat+(1,)])
        points = np.array([list_x, list_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        line = self.colored_line_segments(ax, segments, list_att, cmap, linewidth=0, val_max=val_max, val_min=val_min)
        cbar = fig.colorbar(line, aspect=40, pad=0.02, shrink=0.5)
        cbar.set_label(label='어텐션 가중치 절대값' if language=='kor' else 'Absolute attention weights', rotation=270, labelpad=20)
        cbar.outline.set_visible(False)
        pos = ax.get_position()
        cbar_height = 0.5
        cbar.ax.set_position([pos.x1 + 0.01, pos.y0 + (pos.y1 - pos.y0) * (1 - cbar_height), 0.02, (pos.y1 - pos.y0) * cbar_height])

        # Save
        plt.subplots_adjust(left=-0.02, right=1.0, bottom=0.02, top=1, hspace=0, wspace=0)
        plt.savefig(f'{save_directory}/plot_ecg_attention_{language}_{arrhythmia}.png', dpi=300)
        fig.clf()
        plt.clf()
        plt.close()

    def _plot_circular_attention(self, output_dict, arrhythmia="afib", language="eng", save_directory=None):
        color_heat = self.color_afib if arrhythmia == "afib" else self.color_cia

        att = np.abs(output_dict[f'npy_attention_{arrhythmia}'] * self.att_modifier)
        att_rms = np.sqrt(np.mean(att**2, axis=1))

        tick_labels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        N = len(tick_labels)
        theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        radii = self.norm_encoder(att_rms)
        width = 1.8 * np.pi / N
        color_heat = self.color_set_alpha_one(color_heat)

        colors = [color_heat+(0.1,), color_heat+(0.9,)]
        n_bins = 100
        cmap_name = 'custom_red'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6,4.7))
        bars = ax.bar(theta, radii, width=width, bottom=0, align='center')
        ax.add_patch(plt.Circle((0.5, 0.5), 1, transform=ax.transAxes, color=(1,0,0), alpha=0.02))

        norm = plt.Normalize(np.min(radii), np.max(radii))
        for r, bar in zip(radii, bars):
            bar.set_facecolor(cm(r))

        ax.set_xticks(theta)
        ax.set_xticklabels(tick_labels, fontdict={'fontweight': 'bold'})
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', labelsize=10, color=(0, 0, 0, 0.5))
        ax.grid(alpha=0.2)
        ax.spines['polar'].set_color((0, 0, 0, 0.2))
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.5)
        cbar.set_label(label= '어텐션 가중치의 제곱평균제곱근' if language=='kor' else 'RMS of attention weights', rotation=270, labelpad=20)
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: self.adaptive_format(self.norm_decoder(x, att_rms.min(), att_rms.max()), att_rms.max()-att_rms.min())))
        cbar.outline.set_visible(False)
        pos = ax.get_position()
        cbar_height = 0.5
        cbar.ax.set_position([pos.x1 + 0.1, pos.y0 + (pos.y1 - pos.y0) * (1 - cbar_height), 0.02, (pos.y1 - pos.y0) * cbar_height])

        plt.subplots_adjust(left=0.08, right=0.94, bottom=0, top=1, hspace=0, wspace=0)
        plt.savefig(f'{save_directory}/plt_circular_attention_{language}_{arrhythmia}.png', dpi=300)
        fig.clf()
        plt.clf()
        plt.close()


    def _plot_heatmap(self, output_dict, arrhythmia="afib", language="eng", save_directory=None):
        color_heat = self.color_afib if arrhythmia == "afib" else self.color_cia
        color_low = (255, 243, 154) if arrhythmia == "afib" else (183, 255, 250)

        try:
            if output_dict[f'modified_prob_{arrhythmia}'] is not None:
                prob = float(output_dict[f'modified_prob_{arrhythmia}'])
            else:
                prob = float(output_dict[f'averaged_prob_{arrhythmia}'])
        except ValueError:
            #print(f"Invalid value for prob_{arrhythmia}: {output_dict[f'prob_{arrhythmia}']}")
            prob = 0.0  

        try:
            age = int(output_dict['age'])
        except ValueError:
            #print(f"Invalid value for age: {output_dict['age']}")
            age = 0  

        n_disc = 20
        color_list = [
            self.color_set_alpha_one(color_low),
            self.color_set_alpha_one(color_heat)
        ]
        color_heat = self.color_set_alpha_one(color_heat)
        threshold_list = [0, 1]

        template_sp_valid = self.template_sp[self.template_sp['age']<90]
        disc_age, age_min, age_max = self.discretizer(template_sp_valid['age'], n_disc)
        disc_scr, scr_min, scr_max = self.discretizer(template_sp_valid[f'pred_{arrhythmia}'], n_disc)
        histogram2d = self.count_value_pairs(disc_age, disc_scr, max_value=n_disc-1)
        color_arrays = self.create_color_gradient(n_disc, color_list, threshold_list)

        ticklabel_age = np.round(np.concatenate([age_min, [age_max[-1]]], axis=0)).astype('int')
        ticklabel_scr = np.round(np.concatenate([scr_min, [scr_max[-1]]], axis=0), 2)

        prob_adj = (prob - ticklabel_scr.min()) / (ticklabel_scr.max() - ticklabel_scr.min()) * n_disc
        age_adj = (age - ticklabel_age.min()) / (ticklabel_age.max() - ticklabel_age.min()) * n_disc
        prob_adj = math.floor(prob_adj)
        age_adj = math.floor(age_adj)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim([0, n_disc])
        ax.set_ylim([0, n_disc])

        ax.set_xticks(np.arange(n_disc+1))
        ax.set_xticklabels(ticklabel_age)
        ax.set_yticks(np.arange(n_disc+1))
        ax.set_yticklabels(ticklabel_scr)
        ax.set_xlabel("연령 (년)" if language == "kor" else "Age (year)")
        ax.set_ylabel("심방세동 발생 확률" if language == "kor" else "Probability of AFIB/AFL")

        ax.plot([0, age_adj+0.5], [prob_adj+0.5, prob_adj+0.5], color=color_heat, alpha=0.35, linewidth=6, zorder=2)
        ax.plot([age_adj+0.5, age_adj+0.5], [0, prob_adj+0.5], color=color_heat, alpha=0.35, linewidth=6, zorder=2)

        for i in range(n_disc): 
            for j in range(n_disc): 
                t_alpha = histogram2d[i, j]
                t_color = color_arrays[j]
                rect = FancyBboxPatch((i+0.15, j+0.15), 0.7, 0.7, linewidth=0,
                                    boxstyle="round,pad=0.1",
                                    edgecolor=None, facecolor=t_color, alpha=t_alpha)
                ax.add_patch(rect)

        image_box = OffsetImage(self.macai_button, zoom=0.01)
        ab = AnnotationBbox(image_box, (age_adj+0.5, prob_adj+0.5), frameon=False)
        ax.add_artist(ab)

        cmap = LinearSegmentedColormap.from_list('custom', [x+(0.8,) for x in color_list])
        points = np.array([[0, 0], [1, 1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        line = self.colored_line_segments(ax, segments, segments[0, 1, :], cmap, linewidth=0)
        cbar = fig.colorbar(line, aspect=40, pad=0.01, shrink=1)
        cbar.outline.set_visible(False)
        cbar.set_ticks([0.2, 0.4, 0.6, 0.8])

        plt.subplots_adjust(bottom=0.1, top=0.97, left=0.12, right=1.05, hspace=0, wspace=0)
        plt.savefig(f'{save_directory}/plot_heatmap_{language}_{arrhythmia}.png', dpi=300)
        fig.clf()
        plt.clf()
        plt.close()



    def _plot_hazard_ratio(self, output_dict, language="eng", save_directory=None):
        # prob = output_dict['prob_afib']
        if output_dict['modified_prob_afib'] is not None:
            prob = output_dict['modified_prob_afib']
        else:
            prob = output_dict['averaged_prob_afib']
        sex = output_dict['sex']

        if sex == "Male":
            data = self.template_hr[~self.template_hr.index.isin([1])].reset_index(drop=True)
        else:
            data = self.template_hr[~self.template_hr.index.isin([0])].reset_index(drop=True)

        data['estimated_HR'] = np.exp(np.log(data['HR']) * prob)
        data = data.sort_index(ascending=False).reset_index(drop=True)
        data['Ratio'] = data['estimated_HR']/data['HR']

        log_HR = np.log(data['HR'])
        log_estimated_HR = np.log(data['estimated_HR'])
        colors_dark = [
            (236, 153, 37),
            (218, 88, 38),
            (198, 37, 68),
            (215, 89, 120),
            (143, 111, 156),
            (82, 108, 158),
            (53, 153, 206),
            (117, 177, 118)
        ]
        colors_bright = [
            (254, 196, 20),
            (244, 117, 42),
            (235, 49, 73),
            (243, 127, 150),
            (174, 138, 182),
            (110, 134, 183),
            (77, 187, 236),
            (142, 202, 129)
        ]
        colors_dark = [(r/255, g/255, b/255) for (r,g,b) in colors_dark]
        colors_bright = [(r/255, g/255, b/255) for (r,g,b) in colors_bright]

        bar_width = 0.65
        circle_modifier = 1.6
        dot_modifier_inner = 1.1
        dot_modifier_outer = 1.2
        shadow_modifier = 0.01

        fig, ax = plt.subplots(figsize=(8, 6))
        n_outcomes = len(data[f'Outcome_{language}'])
        index = np.arange(n_outcomes)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        def adjust_function(n_layer, val_start, val_end):
            return np.linspace(val_start, val_end, n_layer)

        for i in range(n_outcomes):
            adj_x = self.cbar_x*bar_width+i
            adj_y_lower = self.cbar_y_lower*bar_width/3*circle_modifier+log_HR[i]
            adj_y_upper = self.cbar_y_upper*bar_width/3*circle_modifier+log_HR[i]
            segments = self.create_segments_vertical(adj_x, adj_y_lower, adj_y_upper)
            tcolor = (0.9, 0.9, 0.9, 1)
            cmap = LinearSegmentedColormap.from_list('custom', [tcolor, tcolor])
            line = self.colored_line_segments(ax, segments, np.repeat(0, adj_x.shape[0]), cmap)
            cbar = fig.colorbar(line)
            cbar.remove()

            adj_x = self.cbar_x*bar_width+i
            adj_y_lower = self.cbar_y_lower*bar_width/3*circle_modifier+log_estimated_HR[i]
            adj_y_upper = self.cbar_y_upper*bar_width/3*circle_modifier+log_estimated_HR[i]
            segments = self.create_segments_vertical(adj_x, adj_y_lower, adj_y_upper)
            tcolor1 = colors_bright[i]; tcolor2 = colors_dark[i]
            cmap = LinearSegmentedColormap.from_list('custom', [(0, tcolor1), (0.5, tcolor1), (0.5, tcolor2), (1, tcolor2)])
            line = self.colored_line_segments(ax, segments, np.concatenate([np.repeat(0, adj_x.shape[0]//2),np.repeat(1, adj_x.shape[0]-adj_x.shape[0]//2)], axis=0), cmap)
            cbar = fig.colorbar(line)
            cbar.remove()

            n_layer = 10
            array_shrinkage = adjust_function(n_layer, val_start=dot_modifier_outer, val_end=dot_modifier_outer*1.075)
            color_shrinkage = adjust_function(n_layer, val_start=0.01, val_end=0.02)

            for (a, c) in zip(array_shrinkage, color_shrinkage):
                adj_x = self.cdot_x*bar_width*a+i+shadow_modifier
                adj_y_lower = self.cdot_y_lower*bar_width/3*circle_modifier*a+log_estimated_HR[i]-shadow_modifier
                adj_y_upper = self.cdot_y_upper*bar_width/3*circle_modifier*a+log_estimated_HR[i]-shadow_modifier
                segments = self.create_segments_vertical(adj_x, adj_y_lower, adj_y_upper)
                tcolor = (0, 0, 0, c)
                cmap = LinearSegmentedColormap.from_list('custom', [tcolor, tcolor])
                line = self.colored_line_segments(ax, segments, np.repeat(0, adj_x.shape[0]), cmap)
                cbar = fig.colorbar(line)
                cbar.remove()

            adj_x = self.cdot_x*bar_width*dot_modifier_outer+i
            adj_y_lower = self.cdot_y_lower*bar_width/3*dot_modifier_outer*circle_modifier+log_estimated_HR[i]
            adj_y_upper = self.cdot_y_upper*bar_width/3*dot_modifier_outer*circle_modifier+log_estimated_HR[i]
            segments = self.create_segments_vertical(adj_x, adj_y_lower, adj_y_upper)
            tcolor = (0.95, 0.95, 0.95, 1)
            cmap = LinearSegmentedColormap.from_list('custom', [tcolor, tcolor])
            line = self.colored_line_segments(ax, segments, np.repeat(0, adj_x.shape[0]), cmap)
            cbar = fig.colorbar(line)
            cbar.remove()

            adj_x = self.cdot_x*bar_width*dot_modifier_inner+i
            adj_y_lower = self.cdot_y_lower*bar_width/3*dot_modifier_inner*circle_modifier+log_estimated_HR[i]
            adj_y_upper = self.cdot_y_upper*bar_width/3*dot_modifier_inner*circle_modifier+log_estimated_HR[i]
            segments = self.create_segments_vertical(adj_x, adj_y_lower, adj_y_upper)
            tcolor = colors_dark[i]
            cmap = LinearSegmentedColormap.from_list('custom', [tcolor, tcolor])
            line = self.colored_line_segments(ax, segments, np.repeat(0, adj_x.shape[0]), cmap)
            cbar = fig.colorbar(line)
            cbar.remove()

            ax.text(i, log_estimated_HR[i], "{0:.2f}".format(data['estimated_HR'][i]),
                    fontsize=13, color=(0.95, 0.95, 0.95), weight="bold",
                    horizontalalignment='center', verticalalignment='center')

        ax.set_xlim(-0.5, n_outcomes-0.5)
        ax.set_xlabel("심방세동 합병증" if language == "kor" else "Patient outcomes")
        ax.set_xticks(index)
        ax.set_xticklabels(data[f'Outcome_{language}'], rotation=90)
        ax.xaxis.set_minor_locator(ticker.NullLocator())

        ax.set_ylim(-0.25, 1.8)
        yticks = np.array([1, 2**(1/2), 2, 2**(3/2), 4, 4*2**(1/2)]) 
        log_yticks = np.log(yticks)
        ytick_labels = [1, round(2**(1/2), 1), 2, round(2**(3/2)), 4, round(4*2**(1/2), 1)] 
        ax.set_yticks(log_yticks)
        ax.set_yticklabels(ytick_labels)
        ax.set_ylabel("위험비 (log scale)" if language == "kor" else "Hazard ratio (log scale)")

        plt.subplots_adjust(bottom=0.4, top=0.95, left=0.1, right=0.95, hspace=0, wspace=0)
        plt.savefig(f'{save_directory}/plot_hazard_ratio_{language}.png', dpi=300)
        fig.clf()
        plt.clf()
        plt.close()


    def run(self, output_dict, save_directory, num_workers=6):
        # Calculate most_important_lead_idx before multiprocessing
        most_important_lead_idx_afib, most_important_lead_name_afib = self.calculate_most_important_lead_info(output_dict, "afib")
        most_important_lead_idx_cia, most_important_lead_name_cia = self.calculate_most_important_lead_info(output_dict, "cia")
        
        plots = [
            (self._plot_ecg_attention, output_dict, "afib", "eng", save_directory),
            (self._plot_ecg_attention, output_dict, "cia", "eng", save_directory),
            (self._plot_ecg_attention_one_lead, output_dict, "afib", "eng", save_directory),
            (self._plot_ecg_attention_one_lead, output_dict, "cia", "eng", save_directory),
            (self._plot_circular_attention, output_dict, "afib", "eng", save_directory),
            (self._plot_circular_attention, output_dict, "cia", "eng", save_directory),
            (self._plot_heatmap, output_dict, "afib", "eng", save_directory),
            (self._plot_heatmap, output_dict, "cia", "eng", save_directory),
            (self._plot_hazard_ratio, output_dict, "eng", save_directory)
        ]

        execution_times = []
        total_start_time = time.time()

        #print(f'Current use_mp mode: {self.use_mp}')
        if self.use_mp:
            #print(f'Working with {num_workers} cores')
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for plot_func, output_dict, *args in plots:
                    start_time = time.time()
                    future = executor.submit(self._plot_worker, plot_func, output_dict, *args)
                    futures.append((future, plot_func.__name__, start_time))

                for future, func_name, start_time in futures:
                    #try:
                    future.result() 
                    end_time = time.time()
                    execution_time = end_time - start_time
                    execution_times.append((func_name, execution_time))
                    #except Exception as e:
                        #print(f"{func_name} failed: {e}")
        else:
            for plot_func, output_dict, *args in plots:
                start_time = time.time()
                #try:
                self._plot_worker(plot_func, output_dict, *args)
                end_time = time.time()
                execution_time = end_time - start_time
                execution_times.append((plot_func.__name__, execution_time))
                #except Exception as e:
                    #print(f"{plot_func.__name__} failed: {e}")

        total_end_time = time.time()
        total_execution_time = total_end_time - total_start_time

        #print("\n각 플롯별 실행 시간:")
        #for func_name, exec_time in execution_times:
            #print(f"{func_name}: {exec_time:.2f}sec")

        #print(f"\n전체 플롯 생성 총 시간: {total_execution_time:.2f}sec")
        
        # Return the most important lead indices and names
        return {
            # 'most_important_lead_idx_afib': most_important_lead_idx_afib,
            'most_important_lead_name_afib': most_important_lead_name_afib,
            # 'most_important_lead_idx_cia': most_important_lead_idx_cia,
            'most_important_lead_name_cia': most_important_lead_name_cia,
            # 'execution_times': execution_times,
            # 'total_execution_time': total_execution_time
        }

    def _plot_worker(self, plot_func, output_dict, *args):
        plot_func(output_dict, *args)


if __name__ == "__main__":
    output_dict = {
        'ecg_orig': np.random.randn(12, 2500),  # 12 leads, 10 seconds of data at 250Hz
        'ecg_filtered': np.random.randn(12, 2500),  # Filtered ECG data
        'npy_attention_afib': np.random.rand(12, 2500),  # Attention weights for AFIB
        'npy_attention_cia': np.random.rand(12, 2500),  # Attention weights for CIA
        'prob_afib': 0.65,  # Example probability for AFIB
        'prob_cia': 0.35,  # Example probability for CIA
        'age': 65,  # Example age
        'sex': 'Male'  # Example sex
    }

    visualizer = ECGVisualizer(path_base="./utils/")
    result = visualizer.run(output_dict, save_directory="./test_output")
    
    print("Most important lead indices and names:")
    print(f"AFIB: {result['most_important_lead_idx_afib']} ({result['most_important_lead_name_afib']})")
    print(f"CIA: {result['most_important_lead_idx_cia']} ({result['most_important_lead_name_cia']})")
