#!/usr/bin/python3

import random
import subprocess

puzzle = 68
h160 = "e0b8a2baee1b77fc703455f39d51477451fc8cfc"

# Constants
LOWER_BOUND = 2 ** (puzzle - 1)
UPPER_BOUND = (2**puzzle) - 1
BIT_GAP = 2**26  # 26-bit gap 

count = 0

while True:
    # Generate the first random number within the valid range
    first_number = random.randrange(LOWER_BOUND, UPPER_BOUND - BIT_GAP)
  
    # Calculate the second number with a 26-bit gap
    second_number = first_number + BIT_GAP
  
    # Ensure the second number is within the upper bound
    if second_number > UPPER_BOUND:
        second_number = UPPER_BOUND  # Set second_number to UPPER_BOUND if it exceeds
  
    # Format both numbers as hexadecimal strings without leading zeros
    first_hex = f"{first_number:X}"  # No leading zeros
    second_hex = f"{second_number:X}"  # No leading zeros
  
    # Prepare the command with the generated random values
    command = [
        "./Cyclone",
        "-h", f"{h160}",
        "-r", f"{first_hex}:{second_hex}"
    ]
  
    # Print the generated numbers
    print(f"Iteration {count + 1}:")
    print(f"First Number (Hex): {first_hex}")
    print(f"Second Number (Hex): {second_hex}")
    print("Executing Cyclone command...")
    print("-" * 50)
  
    # Execute the command and stream output/error to the terminal in real-time
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
  
    # Flag to track if "FOUND MATCH!" has been found
    found_match = False
    lines_after_match = 0
    lines_to_save = []  # List to store lines 1-10

    # Stream stdout and stderr to the terminal
    while True:
        # Read stdout line by line
        stdout_line = process.stdout.readline()
        if stdout_line:
            print(stdout_line, end="")  # Print stdout in real-time
          
            # Check if "FOUND MATCH!" is in the output
            if "FOUND MATCH!" in stdout_line:
                found_match = True
                lines_to_save.append(stdout_line)
          
            # If "FOUND MATCH!"
            if found_match:
                lines_after_match += 1
                if lines_after_match <= 8:
                    lines_to_save.append(stdout_line)
                else:
                    break
        else:
            break
  
    if found_match:
        with open("found_match.txt", "w") as file:
            file.writelines(lines_to_save)  # Write all 9 lines to the file
        print("================== FOUND MATCH! ==================")
        break  # Exit the script after saving the lines
  
    # Wait for the process to complete
    process.wait()
  
    # Check the return code of the process
    if process.returncode != 0:
        print(f"Cyclone command failed with return code: {process.returncode}")
    else:
        print("Cyclone command completed successfully.")
  
    # Increment counter
    count += 1
