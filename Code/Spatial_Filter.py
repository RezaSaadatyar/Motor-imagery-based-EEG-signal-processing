import numpy as np
import matplotlib.pyplot as plt


def spatial_filter(data_filter, position_xy, fs, type_filter, name_channel, display_figure):
    Output_SpatialFilter = np.zeros(np.shape(data_filter))
    N = data_filter.shape[1]
    Time = np.arange(data_filter.shape[0]) / fs
    if type_filter == 'CAR':  # ========= Common Average Reference (CAR) method ========
        Mean = np.mean(data_filter, axis=1)
        for j in range(0, N):
            Output_SpatialFilter[:, j] = data_filter[:, j] - Mean
    else:  # =========== Low or High Laplacian method ===============
        for channel in range(0, N):
            Dis = np.zeros(N)
            for k in range(0, N):  # Euclidean distance
                Dis[k] = np.linalg.norm(position_xy[:, channel] - position_xy[:, k])
            Ind = np.argsort(Dis)  # Sort distance
            if type_filter == 'LL':  # ============= Low Laplacian method ================
                Ind = Ind[1:9]
            elif type_filter == 'HL':  # ============ High Laplacian method ==============
                Ind = Ind[9:21]
            dis = Dis[Ind]
            Dis1 = np.zeros(len(Ind))
            for n in range(0, len(Ind)):  # Euclidean distance for X-axis
                Dis1[n] = np.linalg.norm(position_xy[0, channel] - position_xy[0, Ind[n]])
            Ind1 = np.argsort(Dis1)
            Ind1 = Ind1[0:2]
            for L in range(0, len(Ind)):  # Euclidean distance for Y-axis
                Dis1[L] = np.linalg.norm(position_xy[1, channel] - position_xy[1, Ind[L]])
            Ind2 = np.argsort(Dis1)
            Ind2 = Ind2[0:2]
            In = np.concatenate((Ind1, Ind2), axis=0)
            D = dis[In]
            In = Ind[In]
            W = (1 / D) / sum(1 / D)
            Dis = np.zeros((np.shape(data_filter)[0], 4))
            for r in range(0, 4):
                Dis[:, r] = data_filter[:, In[r]] * W[r]
            Output_SpatialFilter[:, channel] = data_filter[:, channel] - np.sum(Dis, axis=1)
        if display_figure == "On":  # ============== plot system 10-20 EEG ================
            plt.plot(position_xy[0, :], position_xy[1, :], 'ob', markerfacecolor='b', markersize=14)
            plt.plot(position_xy[0, channel], position_xy[1, channel], 'og', markerfacecolor='g', markersize=14)
            plt.plot(position_xy[0, Ind], position_xy[1, Ind], 'or', markerfacecolor='r', markersize=14)
            plt.plot(position_xy[0, Ind[Ind1]], position_xy[1, Ind[Ind1]], 'ok', markerfacecolor='k', markersize=14)
            plt.plot(position_xy[0, Ind[Ind2]], position_xy[1, Ind[Ind2]], 'ok', markerfacecolor='k', markersize=14)
            plt.yticks([])
            plt.xticks([])
            plt.show()
    if display_figure == "On":  # ==== Plot output Spatial Filter and Data (one channel) ====
        N1 = int(data_filter.shape[0] / 3)
        Channel = 1
        plt.subplot(211)
        plt.plot(Time[0:N1], data_filter[0:N1, Channel], label="Raw signal for beta and mu band")
        plt.title("Analyzing channel " + name_channel[Channel] + " using the " + type_filter + " method", color="green")
        plt.legend()
        plt.subplot(212)
        plt.plot(Time[0:N1], Output_SpatialFilter[0:N1, Channel], label="Filtered signal using " + type_filter + " method")
        plt.legend(), plt.xlabel('Time(Sec)')
        plt.show()
    return Output_SpatialFilter
