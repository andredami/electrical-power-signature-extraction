from os import cpu_count
import numpy as np
import pandas as pd
import scipy as sp
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import datetime
import math
#import multiprocessing
from multiprocessing import current_process
from joblib import Parallel, delayed
from tqdm import trange, tqdm
from dtw import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.signal import argrelextrema
from kneed import KneeLocator

def _get_agg_font(self, prop):
    """
    Get the font for text instance t, cacheing for efficiency
    """
    if __debug__: verbose.report('RendererAgg._get_agg_font',
                                 'debug-annoying')

    key = hash(prop)
    key += current_process().pid

    font = RendererAgg._fontd.get(key)

    if font is None:
        fname = findfont(prop)
        #font = RendererAgg._fontd.get(fname)
        if font is None:
            font = FT2Font(
                fname,
                hinting_factor=rcParams['text.hinting_factor'])
            RendererAgg._fontd[fname] = font
        RendererAgg._fontd[key] = font

    font.clear()
    size = prop.get_size_in_points()
    font.set_size(size, self.dpi)

    return font

#num_cores = multiprocessing.cpu_count()
matplotlib.use('agg')

def setupPlt():
    matplotlib.pyplot.switch_backend('Agg')
    plt.rcParams['figure.figsize'] = (20.0, 20.0*9.0/16.0)
    plt.rcParams['figure.dpi'] = 400.0
    plt.rcParams['axes.titlesize'] = 20.0
    plt.rcParams['axes.labelsize'] = 16.0
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8.0
    plt.rcParams['xtick.labelsize'] = 14.0
    plt.rcParams['ytick.labelsize'] = 14.0
    plt.rcParams['legend.fontsize'] = 14.0

DATASET = Path("Datasets/REDD/low_freq/")


def build_house_db(TARGET_HOUSE, DATASET):
    setupPlt()
    house_dir = DATASET / TARGET_HOUSE

    PLOTS = Path("Plots/REDD/low_freq") / TARGET_HOUSE
    PLOTS.mkdir(parents=True, exist_ok=True)

    labels_file = house_dir / "labels.dat"
    channel_files = list(house_dir.glob('channel_*.dat'))

    labels = pd.read_table(labels_file, sep=' ', header=None, index_col=0)

    channels = None
    for __channel_file in channel_files:
        __channel_n = __channel_file.stem.split('_')[1]
        __channel = pd.read_table(__channel_file, sep=' ', names=[
                                  '' + labels[1][int(__channel_n)] + '@' + __channel_n], index_col=0, parse_dates=True, date_parser=lambda t: datetime.datetime.fromtimestamp(int(t)))
        if channels is None:
            channels = __channel
        else:
            channels = pd.concat([channels, __channel], axis=1, join='inner')
    del(__channel, __channel_n, __channel_file)

    TARGET_CHANNEL_NAMES = labels[1].unique()

    selected_channels = labels[labels[1].isin(TARGET_CHANNEL_NAMES)]
    target_channels = (
        selected_channels[1] + '@' + selected_channels.index.astype(str)).tolist()

    def generate_channel_graphs(channel, PLOTS):
        setupPlt()
        channels.plot(y=channel, subplots=False, title=TARGET_HOUSE,
                      xlabel="time", ylabel="power [W]", grid=True)
        path = PLOTS / channel
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / "full_trace.png", facecolor='white')

    def process_single_channel(TARGET_CHANNEL_NAME, channel, PLOTS):
        setupPlt()
        channel_plots = PLOTS / TARGET_CHANNEL_NAME
        channel_plots.mkdir(parents=True, exist_ok=True)

        KDE = sp.stats.gaussian_kde(channel)

        x = np.linspace(0, channel.max(), 2000)
        y = KDE(x)
        min_pos = argrelextrema(y, np.less)
        max_pos = argrelextrema(y, np.greater)

        original_bands = x[min_pos]

        fig = plt.figure()
        fig.subplots_adjust(top=0.8)
        axes = fig.add_subplot(1, 1, 1)
        axes.set_title('Probability Density (Gaussian KDE extimation)')
        axes.set_xlabel('power [W]')
        axes.set_ylabel('probability density')
        axes.plot(x, y, label=TARGET_CHANNEL_NAME)
        axes.scatter(x[min_pos], y[min_pos], s=100,
                     marker='v', c='r', label='local minima')
        axes.scatter(x[max_pos], y[max_pos], s=100,
                     marker='^', c='g', label='local maxima')
        axes.legend()
        axes.minorticks_on()
        axes.grid(which='both')
        plt.savefig(channel_plots / "kde.png", facecolor='white')

        bands = np.unique(
            np.array([channel.min()] + x[min_pos].tolist() + [channel.max()]))

        banded_channel = pd.DataFrame({'Power': channel})
        banded_channel['Band'] = pd.cut(
            banded_channel['Power'], bins=x[min_pos], include_lowest=True)

        band_stats = banded_channel.groupby('Band').describe()
        band_stats

        def aggregate_bands(dataset, column_label, initial_bands_delimiters, joining_threshold=0.01, debug=False):
            setupPlt()
            __max_attempts = len(band_stats) ** 2

            def dprint(*args, **kargs):
                if debug:
                    print(*args, **kargs)

            def __should_aggregate(sourceBand_idx, targetBand_idx, banding_stats):
                dprint("Trying to merge in ", sourceBand_idx)
                if banding_stats.loc[targetBand_idx, 'count'] * joining_threshold <= banding_stats.loc[sourceBand_idx, 'count']:
                    return False
                else:
                    std = banding_stats.loc[sourceBand_idx, 'std']
                    if np.isnan(std):
                        std = 1.0
                    # TODO: refine with further threshold
                    skewness = 3 * \
                        (banding_stats.loc[sourceBand_idx, 'mean'] -
                         banding_stats.loc[sourceBand_idx, '50%']) / std
                    dprint(
                        "Could aggregate. Checking Pearson's alternative coefficient of skewness...", skewness)
                    if targetBand_idx.left == sourceBand_idx.right:
                        return skewness <= 0
                    elif targetBand_idx.right == sourceBand_idx.left:
                        return skewness >= 0
                    else:
                        raise Exception(
                            'Trying to aggregate non continuous band')

            __bands = initial_bands_delimiters.copy()
            __available = np.full(len(__bands) - 1, True)

            for __attempt in range(__max_attempts, 0, -1):
                if __available.any() == False:
                    dprint("No bands to examin. Finished!")
                    break;

                __dataset = dataset.copy()

                __dataset['Band'] = pd.cut(
                    __dataset[column_label], bins=__bands, include_lowest=True)
                __banding_stats = __dataset.groupby('Band').describe()
                __banding_stats.columns = __banding_stats.columns.map(
                    '{0[1]}'.format)

                dprint(__banding_stats)

                __biggest_band_idx = __banding_stats.loc[__available, 'count'].idxmax(
                )
                __biggest_band_pos = __banding_stats.index.get_loc(
                    __biggest_band_idx)

                dprint("Focusing on ", __biggest_band_idx)

                __neighboring_bands = []
                if __biggest_band_pos - 1 >= 0 and __banding_stats.loc[__banding_stats.index[__biggest_band_pos - 1], 'count'] < __banding_stats.loc[__biggest_band_idx, 'count']:
                    __neighboring_bands.append(
                        __banding_stats.index[__biggest_band_pos - 1])
                if __biggest_band_pos + 1 < len(__banding_stats.index) and __banding_stats.loc[__banding_stats.index[__biggest_band_pos + 1], 'count'] < __banding_stats.loc[__biggest_band_idx, 'count']:
                    __neighboring_bands.append(
                        __banding_stats.index[__biggest_band_pos + 1])

                __secundary_neighboring_band = None
                if len(__neighboring_bands) < 1:
                    dprint("No smaller neighbouring bands... skipping.")
                    __available[__biggest_band_pos] = False
                    continue;
                elif len(__neighboring_bands) == 1:
                    __target_neighboring_band = __neighboring_bands[0]
                elif len(__neighboring_bands) == 2:
                    __target_neighboring_band, __secundary_neighboring_band = (__neighboring_bands[0], __neighboring_bands[1]) if __banding_stats.loc[
                                                                               __neighboring_bands[0], 'count'] > __banding_stats.loc[__neighboring_bands[1], 'count'] else (__neighboring_bands[1], __neighboring_bands[0])
                else:
                    raise Exception('Too much neighboring bands!')

                __to_be_merged = None

                if __should_aggregate(__target_neighboring_band, __biggest_band_idx, __banding_stats):
                    __to_be_merged = __target_neighboring_band
                elif __secundary_neighboring_band != None and __should_aggregate(__secundary_neighboring_band, __biggest_band_idx, __banding_stats):
                    __to_be_merged = __secundary_neighboring_band

                if __to_be_merged == None:
                    __available[__biggest_band_pos] = False
                    dprint(__biggest_band_idx,
                           " could not be merged... skipping.")
                else:
                    if __biggest_band_idx.left == __to_be_merged.right:
                        dprint("Merging ", __to_be_merged,
                               " and ", __biggest_band_idx)
                        __bands = __bands[abs(
                            __bands - __biggest_band_idx.left) > 0.01]
                    elif __to_be_merged.left == __biggest_band_idx.right:
                        dprint("Merging ", __biggest_band_idx,
                               " and ", __to_be_merged)
                        __bands = __bands[abs(
                            __bands - __biggest_band_idx.right) > 0.01]
                    else:
                        raise Exception(
                            'Trying to aggregate non continuous band')
                    __available = np.full(len(__bands) - 1, True)
            return __dataset

        #INTERVALS = 20

        #def explore_band_aggregation(channel, bands, INTERVALS, v):
        #    setupPlt()
        #    i = v / INTERVALS
        #    return (i, len(aggregate_bands(pd.DataFrame({'Power': channel}), 'Power', np.array(bands), joining_threshold=i)['Band'].unique()))

        #band_counts = Parallel(n_jobs=num_cores)(delayed(explore_band_aggregation)(channel, bands, INTERVALS, v) for v in tnrange(INTERVALS, desc = 'threshold exploration'))

        band_counts = []
        INTERVALS = 20
        for v in trange(INTERVALS, desc = 'threshold exploration'):
            i = v / INTERVALS 
            band_counts = band_counts + [(i, len(aggregate_bands(pd.DataFrame({'Power': channel}), 'Power', np.array(bands), joining_threshold=i)['Band'].unique()))]

        x, y = zip(*band_counts)

        kneedle = KneeLocator(x, y, curve="convex", direction="decreasing")

        optimal_threshold = kneedle.knee
        optimal_threshold

        x, y = zip(*band_counts)

        fig = plt.figure()
        fig.subplots_adjust(top=0.8)
        axes = fig.add_subplot(1,1,1)
        axes.set_title('Band-merging Threshold Exploration')
        axes.set_xlabel('threshold')
        axes.set_ylabel('number of bands')
        axes.plot(x, y, label=TARGET_CHANNEL_NAME)
        axes.vlines(optimal_threshold, min(map(lambda x: x[1],band_counts)), max(map(lambda x: x[1],band_counts)), linestyles='dashed', colors='red', label='elbow point')
        axes.legend()
        axes.minorticks_on()
        axes.grid(which='both')
        plt.savefig(channel_plots / "elbow_point.png", facecolor='white')
  
        opt_banded_channel = aggregate_bands(pd.DataFrame({'Power': channel}), 'Power', np.array(bands), joining_threshold = optimal_threshold)

        fig = plt.figure()
        fig.subplots_adjust(top=0.8)
        axes = fig.add_subplot(1,1,1)
        axes.set_title('Signal banding (optimized)')
        axes.set_xlabel('time')
        axes.set_ylabel('power [W]')
        axes.plot(opt_banded_channel.index, opt_banded_channel['Power'], label=TARGET_CHANNEL_NAME)
        axes.hlines(original_bands, opt_banded_channel.index.min(), opt_banded_channel.index.max(), linestyles='dashed', colors='gray', label='discarded band limit')
        axes.hlines([x.left for x in opt_banded_channel['Band'].unique()], opt_banded_channel.index.min(), opt_banded_channel.index.max(), colors='r', label='band limit')
        axes.legend()
        axes.grid(which='both')
        plt.savefig(channel_plots / "bands.png", facecolor='white')

        stats_opt_banded_channel = opt_banded_channel.groupby('Band').describe()
        stats_opt_banded_channel


        bands = opt_banded_channel['Band'].dtype
        zero_band = bands.categories[0]
        zero_mean = stats_opt_banded_channel.loc[zero_band, ('Power', 'mean')]
        mean_sampling_period = np.diff(opt_banded_channel.index.values).mean()

        signature_traces = opt_banded_channel.copy()
        signature_traces['Is_Zero'] = signature_traces['Band'].map(lambda x: x == zero_band)
        signature_traces['Candidate_ID'] = (signature_traces['Is_Zero'] != signature_traces['Is_Zero'].shift()).cumsum()
        signature_traces = signature_traces[signature_traces['Is_Zero'].apply(lambda x: not x)]

        band_dict = {val: pos for pos, val in enumerate(signature_traces['Band'].dtype.categories)}

        def compute_sequence(vals):
            res = []
            for val in vals:
                code = band_dict[val]
                if len(res) <= 0 or not res[-1] == code:
                    res.append(code)
            return res

        candidate_signatures = signature_traces.groupby('Candidate_ID')
        candidate_sequences = {}
        for candidate_id, group in candidate_signatures:
            candidate_sequences[candidate_id] = compute_sequence(group['Band'].values)
        candidate_groups = pd.DataFrame([[k, '#'.join(str(x) for x in v)] for k, v in candidate_sequences.items()], columns=['Candidate_ID', 'Code']).groupby('Code')

        sig_channel_plots = channel_plots / "candidates"
        sig_channel_plots.mkdir(parents=True, exist_ok=True)

        def plot_candidate_groups(idx, key, group, sig_channel_plots, mean_sampling_period, zero_mean):
            setupPlt()
            group_channel_plots = sig_channel_plots / str(idx + 1)
            group_channel_plots.mkdir(parents=True, exist_ok=True)
            plt.title('Signature group ' + str(idx + 1) + ' (candidate count: ' + str(len(group)) + ')')
            plt.suptitle(TARGET_CHANNEL_NAME)
            plt.xlabel('duration [s]')
            plt.ylabel('power [W]')
            for id in group['Candidate_ID']:
                candidate_signature = candidate_signatures.get_group(id)
                # duration unit
                start = candidate_signature.index.min()
                end = candidate_signature.index.max()
                candidate_signature.index = [candidate_signature.index - start + mean_sampling_period]
                candidate_signature = pd.concat([pd.DataFrame({'Power': [zero_mean]}, index = [ start - start ]), candidate_signature, pd.DataFrame({'Power': [zero_mean]}, index = [ end - start + 2 * mean_sampling_period ])], axis=0, join='outer',ignore_index=False, copy=True)

                def fix_index(broken_index):
                    try:
                        [idx] = broken_index
                        return idx
                    except TypeError:
                        return broken_index

                candidate_signature.index = [fix_index(v) for v in candidate_signature.index]

                plt.plot([pd.to_timedelta(v).seconds for v in candidate_signature.index.values], candidate_signature['Power'])
            plt.savefig(group_channel_plots / "candidates.png", facecolor='white')
            fig = plt.gcf()
            fig.clf()
            plt.close()

        #Parallel(n_jobs=num_cores)(delayed(plot_candidate_groups)(idx, key, group, sig_channel_plots, mean_sampling_period, zero_mean, group_channel_plots) for (idx, (key, group)) in tqdm(enumerate(candidate_groups), 'groups', total=len(candidate_groups)))

        for (idx, (key, group)) in tqdm(enumerate(candidate_groups), 'Candidate Group Plot', total=len(candidate_groups)):
            plot_candidate_groups(idx, key, group, sig_channel_plots, mean_sampling_period, zero_mean)

        for (idx, (key, group)) in tqdm(enumerate(candidate_groups), 'Candidate Group Analyses', total=len(candidate_groups)):
            target_group = group
            target_candidates = signature_traces.loc[signature_traces['Candidate_ID'].isin(target_group['Candidate_ID'])].groupby('Candidate_ID')

            group_channel_plots = sig_channel_plots / str(idx + 1)
            group_channel_plots.mkdir(parents=True, exist_ok=True)

            candidate_ids = list(target_candidates.groups.keys())

            target_dissimilarity_matrix = np.ndarray((len(target_candidates), len(target_candidates)))

            def fix_index(candidate_signature):
                start = candidate_signature.index.min()
                candidate_signature.index = [pd.to_timedelta(v).seconds for v in [i - start for i in candidate_signature.index.values]]
                return candidate_signature

            if (len(target_candidates) <= 1):
                target_dissimilarity_matrix[0,0] = 0
                minimizing_signature_id = 0
            else:
                for (r, c), _ in tqdm(np.ndenumerate(target_dissimilarity_matrix), 'dissimilarity matrix', total=target_dissimilarity_matrix.size):
                    source = fix_index(target_candidates.get_group(candidate_ids[r]))['Power']
                    reference = fix_index(target_candidates.get_group(candidate_ids[c]))['Power']
                    target_dissimilarity_matrix[r, c] = dtw(source, reference, distance_only=True).distance
                target_dissimilarity_matrix
                minimizing_signature_id = np.argmin(np.sum(target_dissimilarity_matrix, axis=0))

            signature = signature_traces.loc[signature_traces['Candidate_ID'] == candidate_ids[minimizing_signature_id]]

            signature_traces.loc[signature_traces['Candidate_ID'] == candidate_ids[minimizing_signature_id], 'IsSignature'] = numpy.ones(len(signature), dtype=bool)

            sigfig = plt.figure()

            plt.suptitle(TARGET_CHANNEL_NAME)
            plt.title('Signature ' + str(idx + 1))
            plt.xlabel('duration [s]')
            plt.ylabel('power [W]')
            # duration unit
            start = signature.index.min()
            end = signature.index.max()
            signature.index = [signature.index - start + mean_sampling_period]
            signature = pd.concat([pd.DataFrame({'Power': [zero_mean]}, index = [ start - start ]), signature, pd.DataFrame({'Power': [zero_mean]}, index = [ end - start + 2 * mean_sampling_period ])], axis=0, join='outer',ignore_index=False, copy=True)

            def fix_index(broken_index):
                try:
                    [idx] = broken_index
                    return idx
                except TypeError:
                    return broken_index

            signature.index = [fix_index(v) for v in signature.index]

            plt.plot([pd.to_timedelta(v).seconds for v in signature.index.values], signature['Power'])
            plt.savefig(group_channel_plots / "signature.png", facecolor='white')
            fig = plt.gcf()
            fig.clf()
            plt.close()

            if len(target_candidates) >= 1:
                dmfig, ax = plt.subplots()
                plt.suptitle(TARGET_CHANNEL_NAME)
                im = ax.pcolor(target_dissimilarity_matrix)

                # We want to show all ticks...
                ax.set_xticks(np.arange(len(candidate_ids)))
                ax.set_yticks(np.arange(len(candidate_ids)))
                # ... and label them with the respective list entries
                ax.set_xticklabels(candidate_ids)
                ax.set_yticklabels(candidate_ids)

                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
                        rotation_mode="anchor")

                # Loop over data dimensions and create text annotations.
                # for i in tnrange(len(candidate_ids)):
                #    for j in range(len(candidate_ids)):
                #        text = ax.text(j, i, target_dissimilarity_matrix[i, j],
                #                       ha="center", va="center", color="w")

                ax.set_title("Dissimilarity matrix (candidate group " + str(idx + 1) + ")")
                dmfig.tight_layout()
                dmfig.colorbar(im)
                plt.savefig(group_channel_plots / "dissimilarity.png", facecolor='white')
                fig = plt.gcf()
                fig.clf()
                plt.close()

        signature_cadidate_ids = signature_traces.loc[signature_traces['IsSignature'] == True,'Candidate_ID'].unique()

        signatures = signature_traces.loc[signature_traces['Candidate_ID'].isin(signature_cadidate_ids)].groupby('Candidate_ID')

        def fix_index(candidate_signature):
            start = candidate_signature.index.min()
            candidate_signature.index = [pd.to_timedelta(v).seconds for v in [i - start for i in candidate_signature.index.values]]
            return candidate_signature

        def dwt_dist(candidate_id_x, candidate_id_y):
            source = fix_index(signatures.get_group(int(candidate_id_y)))['Power']
            reference = fix_index(signatures.get_group(int(candidate_id_x)))['Power']
            return float(dtw(source, reference, distance_only=True).distance)

        def len_dist(candidate_id_x, candidate_id_y):
            source = fix_index(signatures.get_group(int(candidate_id_y)))['Power']
            reference = fix_index(signatures.get_group(int(candidate_id_x)))['Power']
            return float(abs(len(source) - len(reference)))

        X = np.array(signature_cadidate_ids).reshape(-1, 1)

        def normalize(array):
            return (array - np.min(array)) / np.ptp(array)

        m1 = normalize(pairwise_distances(X, X, metric=dwt_dist))
        m2 = normalize(pairwise_distances(X, X, metric=len_dist))
        m =  0.4*m1 + 0.6*m2

        s = list()

        for nc in tqdm(range(2, len(X)), 'clusters'):
            agg = AgglomerativeClustering(n_clusters=nc, affinity='precomputed',
                                    linkage='average')

            u = agg.fit_predict(m)
            s.append(silhouette_score(m, u, metric="precomputed"))

        plt.plot(list(range(2, len(X))), s)

        optimal_threshold_sig = np.argmax(s) + 2

        agg = AgglomerativeClustering(n_clusters=optimal_threshold_sig, affinity='precomputed',
                                    linkage='average')
        u = agg.fit_predict(m)
        u

        definitive_signatures_candidate_id = list()

        for cluster in range(0, optimal_threshold_sig):
            cond = u == cluster
            group = np.array(m1)[cond][:, cond]
            best = np.argmin(group.sum(axis=0))
            idx = np.searchsorted(np.cumsum(cond), best + 1)
            candidate_id = signature_cadidate_ids[idx]
            definitive_signatures_candidate_id.append(candidate_id)

        definitive_signatures = signature_traces.loc[signature_traces['Candidate_ID'].isin(definitive_signatures_candidate_id)].groupby('Candidate_ID')

        plt.suptitle(TARGET_CHANNEL_NAME)
        plt.title('Signatures')
        plt.xlabel('duration [s]')
        plt.ylabel('power [W]')

        signature_db = channel_plots / "signatures"
        signature_db.mkdir(parents=True, exist_ok=True)

        for (idx, (key, group)) in tqdm(enumerate(definitive_signatures), 'groups', total=len(definitive_signatures)): 

            definitive_signature = group

            start = definitive_signature.index.min()
            end = definitive_signature.index.max()
            definitive_signature.index = [definitive_signature.index - start + mean_sampling_period]
            definitive_signature = pd.concat([pd.DataFrame({'Power': [zero_mean]}, index = [ start - start ]), definitive_signature], axis=0, join='outer',ignore_index=False, copy=True)

            def fix_index(broken_index):
                try:
                    [idx] = broken_index
                    return idx
                except TypeError:
                    return broken_index

            definitive_signature.index = [fix_index(v) for v in definitive_signature.index]

            resampled_signature = definitive_signature['Power'].resample('10S').ffill(limit=1).interpolate()
            resampled_signature = resampled_signature.append(pd.Series([0], index=[resampled_signature.index.max() + resampled_signature.index[1] - resampled_signature.index[0]]))

            plt.plot([pd.to_timedelta(v).seconds for v in resampled_signature.index.values], resampled_signature)
            np.savetxt(signature_db / (str(idx) + '.dat'), resampled_signature.to_numpy(), delimiter=",")


        plt.savefig(channel_plots / "signatures.png", facecolor='white')

    #Parallel(n_jobs=num_cores)(delayed(generate_channel_graphs)(channel, PLOTS) for channel in target_channels)

    for channel in tqdm(target_channels, 'Channel Plots (' + TARGET_HOUSE +')'):
        generate_channel_graphs(channel, PLOTS)


    selected_channels = labels[~labels[1].isin(["mains", "kitchen_outlets", "bathroom_gfi", "outlets_unknown", "miscellaeneous", "subpanel"])]
    target_channels = (
        selected_channels[1] + '@' + selected_channels.index.astype(str)).tolist()

    #Parallel(n_jobs=num_cores)(delayed(process_single_channel)(TARGET_CHANNEL_NAME, channels[TARGET_CHANNEL_NAME], PLOTS) for TARGET_CHANNEL_NAME in target_channels)

    for TARGET_CHANNEL_NAME in tqdm(target_channels, 'Channel Analyses (' + TARGET_HOUSE +')'):
        process_single_channel(TARGET_CHANNEL_NAME, channels[TARGET_CHANNEL_NAME], PLOTS)

import sys

if __name__ == "__main__":
    #Parallel(n_jobs=num_cores)(delayed(build_house_db)(house, DATASET) for house in map(lambda p: p.name, DATASET.glob("*")))
    #houses = list(map(lambda p: p.name, DATASET.glob("*")))
    #for house in tqdm(houses, 'houses'):
    #    build_house_db(house, DATASET)
    print(sys.argv[1])
    build_house_db(sys.argv[1], DATASET)