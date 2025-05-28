import matplotlib.pyplot as plt
import time
import random
import matplotlib 

class PlottingSession:
    def __init__(self, title= 'Velocity Flash Performance Test using IMU PD Feedback', xlabel='Time (s)', ylabel='Value'):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.time_data = []
        self.sensor_data = []
        self.setpoint_data = []
        self.anglecommand_data = []
        self._start_time = None
        self.is_collecting = False

    def start_collection(self):
        print("Starting data collection...")
        self.time_data.clear()
        self.sensor_data.clear()
        self.setpoint_data.clear()
        self.anglecommand_data.clear()
        self._start_time = time.monotonic()
        self.is_collecting = True

    def add_data(self, sensor_value, setpoint_value, angle_value):
        if not self.is_collecting: return
        if self._start_time is None:
             self.is_collecting = False
             return
        current_time = time.monotonic() - self._start_time
        self.time_data.append(current_time)
        self.sensor_data.append(sensor_value)
        self.setpoint_data.append(setpoint_value)
        self.anglecommand_data.append(-angle_value)

    def stop_collection_and_plot(self):
        if not self.is_collecting:
            return

        self.is_collecting = False
        print("Stopped data collection.")

        if not self.time_data:
            print("No data collected to plot.")
            return

        print("Generating plot and waiting for close...")

        fig = None 
        try:
            fig, ax = plt.subplots()
            ax.plot(self.time_data, self.sensor_data, 'r-', label='IMU Sensor Angle')
            ax.plot(self.time_data, self.setpoint_data, 'b-', label='Angle Setpoint')
            ax.plot(self.time_data, self.anglecommand_data, 'g--', label = 'Steer Command to Hoverboard')
            ax.set_title(self.title)
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)
            ax.legend()
            ax.grid(True)
            ax.relim()
            ax.autoscale_view(tight=False, scalex=True, scaley=True)

            plt.show(block=True)
            print("Plot window closed by user. Continuing...")

        except Exception as e:
            print("Error during plotting or showing")
            if fig is not None and plt.fignum_exists(fig.number):
                plt.close(fig)




