import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import numpy as np
import random

def create_schedule(num_animals, start_time_input, num_trials, trial_durations, time_between_animals):
    # Create a list to store the schedule
    start_time_original = convert_time_to_minutes(start_time_input)
    schedule = []
    timeline_data = []  # Data for generating the timeline

    # Initialize the current time
    current_time = start_time_original


    for animal in range(1, num_animals + 1):
        for trial in range(1, num_trials + 1):
            trial_duration = trial_durations[trial - 1]
            

            # Calculate the stagger offset for the first trial of each animal
            if trial == 1:
                trial_offset = (time_between_animals+5) * (animal-1)  # 10 minutes for the intertrial interval
                
                start_time = start_time_original + trial_offset  
            else:
                start_time=current_time

            end_time = start_time + trial_duration

            schedule.append([animal, trial, start_time, end_time])
            timeline_data.append((f"Animal {animal}", trial, start_time, end_time))

            # Update the current time for the next trial
            current_time = end_time

            # Add the intertrial interval after each trial, except for the last one
            if trial < num_trials:
                current_time += 5  # Fixed 10-minute intertrial interval


    # Create a DataFrame for the schedule
    schedule_df = pd.DataFrame(schedule, columns=["Animal", "Trial", "Start Time (min)", "End Time (min)"])

    # Format the times in "hh:mm AM/PM" format
    schedule_df['Start Time (min)'] = schedule_df['Start Time (min)'].apply(convert_minutes_to_time)
    schedule_df['End Time (min)'] = schedule_df['End Time (min)'].apply(convert_minutes_to_time)

    return schedule_df, timeline_data


# Function to convert minutes to "hh:mm AM/PM" format
def convert_minutes_to_time(minutes):
    hours, minutes = divmod(minutes, 60)
    if hours < 0:
        hours += 24
    elif hours >= 24:
        hours -= 24

    am_pm = "AM" if 0 <= hours < 12 else "PM"
    if hours == 0:
        hours = 12
    elif hours > 12:
        hours -= 12

    time_str = f"{hours:02}:{minutes:02} {am_pm}"
    return time_str

# Function to convert a time of day to minutes
def convert_time_to_minutes(time_str):
    try:
        time_obj = datetime.strptime(time_str, '%I:%M %p')
        return time_obj.hour * 60 + time_obj.minute
    except ValueError:
        raise ValueError("Invalid time format. Please use the format 'hh:mm AM/PM' (e.g., '10:30 AM').")


def generate_gantt_chart(timeline_data, num_animals):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set(title="Gantt Chart")
    #ax.set_yticks(range(1, num_animals + 1))
    animal_colors = [plt.cm.tab20(i) for i in range(num_animals)]  # Generate unique colors for animals
    y_ticks = []

    for item in timeline_data:
        animal, trial, start_time, end_time = item  # Unpack the values

        # Convert start_time and end_time to "hh:mm AM/PM" format
        start_time_str = convert_minutes_to_time(start_time)
        end_time_str = convert_minutes_to_time(end_time)

        # Create a label with Animal and Trial information
        label = f"{animal}, Trial {trial}"
        animal_number = int(''.join(filter(str.isdigit, animal)))
        y_ticks.append(animal_number-1)
        # Assign a unique color to each animal
        color = animal_colors[animal_number - 1]

        # Plot the trial as a Gantt bar with transparency
        if trial == 1:
            alpha =0.3
        elif trial ==2:
            alpha =0.5
        elif trial ==3:
            alpha = 0.7
        ax.barh(f"{animal}", left=start_time, width=end_time - start_time, color=color, alpha=alpha, label=label)
        
        # Add labels at the beginning and end of the bar
        
        if trial ==1:
            ax.text(start_time, f"{animal}", f"Start:\n{start_time_str}", ha="left", va="center", fontsize=8)
            ax.text(end_time, f"{animal}", f"End:\n{end_time_str}", ha="right", va="center", fontsize=8)
        elif trial == 2:
            ax.text(start_time, f"{animal}", f"Start:\n{start_time_str}", ha="left", va="bottom", fontsize=8)
            ax.text(end_time, f"{animal}", f"End:\n{end_time_str}", ha="right", va="top", fontsize=8)
        elif trial ==3: 
            ax.text(start_time, f"{animal}", f"Start:{start_time_str}", ha="left", va="bottom", fontsize=8)
            ax.text(end_time, f"{animal}", f"End:{end_time_str}", ha="left", va="top", fontsize=8)
    animal_labels = list(range(1, num_animals + 1))
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.set_yticks([])  # Remove y ticks
    ax.set_xticks([])  # Remove x ticks
    ax.set_yticks(y_ticks)  # Set y-ticks to animal IDs
    #ax.set_yticklabels(animal_labels)  # Set y-tick labels to animal IDs
    plt.show()