"""
   parsed for different format
   raw inspirwed from https://github.com/wojdyr/xylib
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm, colorbar, colors
import scipy.interpolate as si
import matplotlib as mpl


class XRDfile(object):
    """class for import of xrd file


    Attr:
        debug        : if one stop the reading
        file_status  : file status active done..
        version      : version of the orginal file
        data         : list with the scans
        merged       : list of x,y merged

    Methods:
        export
        plot
        plot_info
        print_info
        merge

    available format:
            brucker raw V3 V4

    """

    def __init__(self, filename=None):
        # +super(XRDfile, self).__init__()
        self.info = {}
        read_raw(self, filename)
        self.theta = np.array([i['info']['START_Theta'] for i in self.data])
        self.Dtheta = calc_x(self.data[0]['info']['START_2Theta'],
                             self.data[0]['info']['STEPSIZE'],
                             self.data[0]['counts'])
        self.counts = np.vstack([i['counts'] for i in self.data]).T

    def plot_rsm(self, log=True, levels=30, vmax=None, vmin=None,
                 output=False, **kargs):
        """ plot rms
        paramers
            log (bool): if data to show log
            levels (int): number of levels
            vmax  (float) : maxim value reperesented
            vmin  (float) : min value reperesented
        """
        X, Y = np.meshgrid(self.theta, self.Dtheta)

        intensity = np.copy(self.counts)
        if vmax:
            intensity = np.where(intensity > vmax, vmax, intensity)
        else:
            vmax = intensity.max()
        if vmin:
            intensity = np.where(intensity < vmin, vmin, intensity)
        else:
            vmin = intensity.min()
        if log:
            intensity = np.log10(intensity)
            vmax = np.log10(vmax)
            vmin = np.log10(vmin)
            cbarlabel = 'log(counts)'
        else:
            cbarlabel = 'counts'

        if output:
            return X, Y, intensity, cbarlabel, self.theta, self.Dtheta, self.counts

        self.figure = plt.figure()
        plt.contourf(X, Y, intensity, levels, **kargs)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        plt.xlabel(r'$\theta$($\degree$)')
        plt.ylabel(r'$2\theta$($\degree$)')
        plt.title(f"{self.info['LAYER']}/{self.info['SUBSTRATE']}")

        func = si.RectBivariateSpline(self.theta, self.Dtheta,  self.counts.T)
        def fmt(x, y):
            z = np.take(func(x, y), 0)
            return 'x={x:.5f}  y={y:.5f}  counts={z:.5f}'.format(x=x, y=y, z=z)
        plt.gca().format_coord = fmt

        #ax, _ = mpl.colorbar.make_axes(plt.gca())
        #mpl.colorbar.ColorbarBase(ax, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        #ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        plt.show()


    def plot_rsmQ(self, log=True, vmax=None, vmin=None, levels=30, 
                  output=False, **kargs):
        """ plot rms
        paramers
            theta (np.array 1D): array with theta in degrees
            Dtheta (np.array 1D): array with 2 theta in degrees
            intensity (np array shape=theta.shape, Dtheta.shape)
            log (bool): if data to show log
            levels (int): number of levels
            vmax  (float) : maxim value reperesented
            vmin  (float) : min value reperesented
        """
        DpiWL = 2 * np.pi / self.data[0]['info']['RANGE_WL']
        mesh_rtheta, mesh_rDtheta = np.meshgrid(np.radians(self.theta),
                                                np.radians(self.Dtheta))
        qpara = DpiWL * (np.cos(mesh_rDtheta-mesh_rtheta) - np.cos(mesh_rtheta)) * 1000
        qperp = DpiWL * (np.sin(mesh_rDtheta-mesh_rtheta) + np.sin(mesh_rtheta)) * 1000

        intensity = np.copy(self.counts)
        if vmax:
            intensity = np.where(intensity > vmax, vmax, intensity)
        else:
            vmax = intensity.max()
        if vmin:
            intensity = np.where(intensity < vmin, vmin, intensity)
        else:
            vmin = intensity.min()
        if log:
            intensity = np.log10(intensity)
            vmax = np.log10(vmax)
            vmin = np.log10(vmin)
            cbarlabel = 'log(counts)'
        else:
            cbarlabel = 'counts'

        if output:
            return qpara, qperp, intensity, cbarlabel

        self.figure = plt.figure()
        plt.contourf(qpara, qperp, intensity, levels, **kargs)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        plt.xlabel(r'$q\parallel$($\AA^{-1}$)')
        plt.ylabel(r'$q\perp$($\AA^{-1}$)')
        plt.title(f"{self.info['LAYER']}/{self.info['SUBSTRATE']}")
        plt.tight_layout()
        plt.show()



def calc_x(start, step, data):
    start, step, = float(start), float(step)
    return np.array([start + step * i for i in np.arange(len(data))])


def read_raw(self, filename):
    read_i = lambda arr : np.frombuffer(bfile[arr:], dtype=np.uint32, count=1)[0] # noqa
    read_d = lambda arr : np.frombuffer(bfile[arr:], dtype=np.float64, count=1)[0] # noqa
    read_f = lambda arr : np.frombuffer(bfile[arr:], dtype=np.float32, count=1)[0] # noqa
    read_s = lambda arr, nby: bfile[arr:arr+nby].decode('ascii').rstrip('\x00') # noqa

    self.data = []
    if isinstance(filename, memoryview):
        bfile = filename.tobytes()
    if isinstance(filename, str):
        with open(filename, 'rb') as filex:
            bfile = filex.read()


    head = bfile[:4].decode('ascii')

    if head == "RAW ":
        self.version = "ver. 1"
    elif head == "RAW2":
        self.version = "ver. 2"
    elif head == "RAW1":
        subv = filex.read(3).decode('ascii')
        if subv == ".01":
            self.version = "ver. 3"
    elif (head == "RAW4"):
        self.version = "ver. 4"
    else:
        print("no format detected")
    print('head= ', head)


    if self.version == "ver. 4":
        self.info['Version']= read_s(0, 7)   # version complet
        # file_status=read_uint32_le()
        # status=["done", "active", "aborted", "interrupted"]
        # self.file_status=status[file_status]
        self.info["MEASURE_DATE"] = read_s(12, 12)        # address 12# noqa E510
        self.info["MEASURE_TIME"] = read_s(24, 10)        # address 24->34# noqa E510
        self.info['SCAN_cnt_0'] = read_i(36)      # noqa
        self.info['SCAN_cnt']   = read_i(40)      # noqa
        self.info['SCAN_cnt2']  = read_i(44)      # noqa
        self.info['RUNTIME']    = read_f(48)      # maybe done address 48 52# noqa E510
        self.info['RUNTIME2']   = read_f(48)      # maybe done address 52 56# noqa E510
        #future_info  = read_i()                 # maybe done val 1409  address 56 60   # noqa E510
        #filex.read(3)                                   # maybe done  address 60 61                  # noqa E510

        pos = 61
        while True:
            code = read_i(pos)                                 # offset x->x+4  # noqa E510
            if code in [0, 160]:
                break
            len_seg =  read_i(pos+4)                         # offset 4->8  # noqa E510
            if code == 10:
                key = read_s(pos + 12, 24)
                self.info[key] = read_s(pos+36, len_seg - 36 ) # noqa E510
                pos += len_seg
            elif code == 30:
                self.info["V4_INF_STAGE_CODE"]    = read_i(pos+12)    # noqa
                self.info['GONIOMETER_RADIUS']    = read_f(pos+24)/2  # noqa
                self.info["V4_INF_FIXED_DIVSLIT"] = read_f(pos+36)    # noqa
                self.info["FIXED_ANTISLIT"]       = read_f(pos+60)    # noqa
                self.info["FIXED_DSLIT"]          = read_f(pos+64)    # noqa
                self.info["ALPHA_AVERAGE"]        = read_d(pos+72)    # noqa
                self.info["ALPHA1"]               = read_d(pos+80)    # noqa
                self.info["ALPHA2"]               = read_d(pos+88)    # noqa
                self.info["BETA"]                 = read_d(pos+96)    # noqa
                self.info["ALPHA_RATIO"]          = read_d(pos+104)   # noqa
                self.info["Anode"]           =  read_s(pos+116, 4)    # noqa
                self.info["W_UNIT"]          =  read_s(pos+120, 4)    # noqa
                pos += len_seg
            elif code == 60:
                pos += len_seg
            elif code == 5:
                pos += len_seg
            else:
                print('unknown code', code)
                raise ValueError('unknown code')

        for cur_range in range(self.info['SCAN_cnt']):
            # print(cur_range, pos)
            blkmeta = {}
            blkmeta['SCAN_step1']         = read_i(pos+4)    # noqa
            blkmeta['SCAN_step2']         = read_i(pos+8)    # noqa
            blkmeta['ADDITIONALDETECTOR'] = read_i(pos+24)   # noqa                
            blkmeta['SCAN_type']        = read_s(pos+32, 24) # address 32@@@@@@@@@@ # noqa
            # print filex.tell(), 'should be 642+68 710'
            blkmeta["TIMESTARTED"]     = read_f(pos+68)       # noqa
            blkmeta["START"]           = read_d(pos+72)       # not sure result=10 # noqa
            blkmeta["STEPSIZE"]        = read_d(pos+80)       # 88 # noqa
            blkmeta["STEPS"]           = read_i(pos+88)       # address 88@@@@@@@@@ # noqa
            blkmeta["STEPTIME"]        = read_f(pos+92)       # address 92@@@@@@@@@@ # noqa
            blkmeta["KV"]              = read_f(pos+100)              # address 100@@@@@@@@@ # noqa
            blkmeta["MA"]              = read_f(pos+104)              # address 104@@@@@@@@@ # noqa
            blkmeta["RANGE_WL"]        = read_d(pos+112)              # address 112@@@@@@@@@ # noqa
            datum_size      = read_i(pos+136)                         # address 136@@@@@@@@#  equal to 0??? # noqa
            hdr_size        = read_i(pos+140)                         # address 140@@@@@@@@#  equal to 0??? # noqa
            pos += 160

            next_data = pos + hdr_size
            # print(pos, 'next_data', next_data)
            while hdr_size > 0:
                code    = read_i(pos)        # noqa E510
                len_seg = read_i(pos+4)      # noqa E510
                if code == 10:
                    key = read_s(pos + 12, 24)
                    self.info[key] = read_s(pos+36, len_seg - 36 ) # noqa E510
                    pos += len_seg
                    hdr_size -= len_seg
                elif code == 110:
                    blkmeta["PSD2THETA"]        = read_d(pos+8)              # address 8 @@@@@@@@@@@ # noqa
                    blkmeta["PSDCHANNEL1"]      = read_i(pos+16)          # address 16 @@@@@@@@@@@ # noqa
                    blkmeta["PSDAPERTURE"]      = read_f(pos+20)             # address 20 @@@@@@@@@@@ # noqa
                    blkmeta["PSDTYPE"]          = read_i(pos+24)          # address 24 @@@@@@@@@@@ # noqa
                    blkmeta["PSDFIXED"]         = read_f(pos+4)             # address 28 @@@@@@@@@@@ # noqa 
                    pos += len_seg
                    hdr_size -= len_seg
                elif code == 50:
                    # int 2 
                    un = read_i(pos + 8)
                    key  = read_s(pos+12, 24)                               # address 12 @@@@@@@@@@@ # noqa
                    un2 = read_i(pos + 52)
                    blkmeta[f"START_{key}"]      = read_d(pos+56)          # address 56 @@@@@@@@@@@ # noqa
                    pos += len_seg
                    hdr_size -= len_seg
                elif code == 300:
                    blkmeta["HRXRD"] =[]                                       # address 8 @@@@@@@@@@@ # noqa
                    blkmeta["HRXRD"].append(read_s(pos + 8, 24))
                    blkmeta["HRXRD"].append(read_d(pos + 52))
                    blkmeta["HRXRD"].append(read_d(pos + 60))
                    blkmeta["HRXRD"].append(read_d(pos + 80))
                    blkmeta["HRXRD"].append(read_d(pos + 88))
                    blkmeta["HRXRD"].append(read_d(pos + 96))
                    blkmeta["HRXRD"].append(read_d(pos + 104))
                    blkmeta["HRXRD"].append(read_i(pos + 112))
                    blkmeta["HRXRD"].append(read_d(pos + 116))
                    blkmeta["HRXRD"].append(read_d(pos + 124))
                    blkmeta["HRXRD"].append(read_d(pos + 132))
                    blkmeta["HRXRD"].append(read_d(pos + 204))
                    blkmeta["HRXRD"].append(read_d(pos + 212))
                    blkmeta["HRXRD"].append(read_d(pos + 220))
                    pos += len_seg
                    hdr_size -= len_seg
                else:
                    print('unknown code', code)
                    raise ValueError('unknown code')
            pass
            assert(datum_size == 4)
            y = np.frombuffer(bfile[next_data:], dtype=np.float32, count=blkmeta["STEPS"])
            # x = calc_x(blkmeta["START"], blkmeta["STEPSIZE"], y)
            # print len(x), len(y)
            # print 'shape',x.shape , y.shape
            self.data.append({'counts': y,
                              'info': blkmeta})
            pos += datum_size * blkmeta["STEPS"]

    for j, i in enumerate(self.data):
        i['info']['index'] = j


if __name__ == '__main__':
    import tkinter as tk
    from tkinter import filedialog
    import matplotlib
    matplotlib.use('QtAgg')

    rec_map = [None]

    def browse_file():
        file_path = filedialog.askopenfilename()
        file_path_var.set(file_path)
        # Extract and display just the filename
        rec_map[0] = XRDfile(file_path)
        filename_label_var.set(f"Selected File: {file_path.split('/')[-1]}")

    def plot_rsm():
        rec_map[0].plot_rsm(log=int(CheckVar1.get()))

    def plot_rsmQ():
        rec_map[0].plot_rsmQ(log=int(CheckVar2.get()))

    # Create the main window
    root = tk.Tk()
    root.title("File Browser and Buttons")

    # Variable to store the selected file path
    file_path_var = tk.StringVar()
    # Variable to store the filename for display
    filename_label_var = tk.StringVar()

    # Create a label to display the selected file path
    #file_label = tk.Label(root, textvariable=file_path_var, wraplength=400)
    #file_label.pack(pady=10)

    # Create a label to display just the filename
    filename_label = tk.Label(root, textvariable=filename_label_var)
    filename_label.pack(side=tk.TOP, pady=5)

    # Create a button to browse for a file
    browse_button = tk.Button(root, text="Browse", command=browse_file, width=15)
    browse_button.pack(side=tk.TOP, pady=5)

    # Create a frame to hold the buttons and checkboxes
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.TOP, pady=10)

    # Create a button to process the selected file
    rsm_button = tk.Button(button_frame, text="plot_rsm", command=plot_rsm, width=15)
    rsm_button.grid(row=0, column=0, pady=10)

    # Create a checkbox to the right of the Browse button
    CheckVar1 = tk.IntVar(value=1)
    rsm_checkbox = tk.Checkbutton(button_frame, text="log", variable=CheckVar1)
    rsm_checkbox.grid(row=0, column=1, pady=10)

    # Additional button (you can customize its functionality)
    rsmQ_button = tk.Button(button_frame, text="plot_rsmQ", command=plot_rsmQ, width=15)
    rsmQ_button.grid(row=1, column=0, pady=10)

    # Create a checkbox to the right of the Browse button
    CheckVar2 = tk.IntVar(value=1)
    rsmQ_checkbox = tk.Checkbutton(button_frame, text="log", variable=CheckVar2)
    rsmQ_checkbox.grid(row=1, column=1, pady=10)

    # Run the Tkinter event loop
    root.mainloop()