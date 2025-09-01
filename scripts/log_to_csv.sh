#!/bin/bash
# Script to extract E (epoch) and L (loss) from a variable containing training logs

output_to_csv () {
    # Write CSV header to file
    echo "Epoch,Loss" > "$2_output.csv"

    # Process the variable and append to file
    echo "$1" | awk -F',' '{
        match($2, /E:([0-9]+)/, e)
        match($4, /L:([0-9.]+)/, l)
        if (e[1] != "" && l[1] != "")
            print e[1] "," l[1]
    }' >> "$2_output.csv"
}

log="$(grep -e Train ../artifacts/log.txt)"
output_to_csv "$log" "train"
log="$(grep -e Valid ../artifacts/log.txt)"
output_to_csv "$log" "valid"


