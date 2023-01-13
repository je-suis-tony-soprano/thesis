events = read.csv('C:/Users/35383/Documents/Master/Thesis/data/sequences.csv')

# Use the table function to create a frequency table of the values in the "sequence number" column
sequence_counts <- table(events$sequence)

# Use the which function to identify the rows with values that have a frequency greater than 1
duplicated_rows <- which(sequence_counts > 1)

# Subset the data frame to remove rows with unique values
events_filtered <- events[events$sequence %in% duplicated_rows, ]

write.csv(events_filtered, "C:\\Users\\35383\\Documents\\Master\\Thesis\\data\\events_nlp.csv", row.names=FALSE)

# Use slice to grab rows relating to specific actions at the start and end of sequences
first_action <- events_filtered %>% group_by(sequence) %>%
  slice(1)

last_action <- events_filtered %>% group_by(sequence) %>%
  slice_tail()

second_last_action <- events_filtered %>% group_by(sequence) %>%
  slice(n()-1)

# Creating the feature set
features <- events_filtered %>% group_by(sequence) %>%
  summarise(number_of_passes = sum(type.name == "Pass"),
            number_of_dribbles = sum(type.name == "Dribble"),
            number_of_carries = sum(type.name == "Carry"),
            attack_speed = mean(duration),
            shot = sum(type.name == "Shot"))

features$start_distance_x <- 120 - first_action$location.x
features$end_distance_x <- 120 - last_action$location.x
features$end_distance_x_2 <- 120 - second_last_action$location.x

features <- features %>%
  mutate(start_offcentre = abs(first_action$location.y - 40))
  
features <- features %>%
  mutate(end_offcentre = abs(last_action$location.y - 40))

# Add original start_location_y
features$start_location_y <- first_action$location.y

# General tidying removing extra shots and sequences from corners
clean_shots <- replace(features$shot, features$shot>1, 1)
features$shot <- clean_shots

features <- features %>%
  filter(start_distance_x != 0)

# Rearrange the columns into a more intuitive order
features <- features[, c(1,2,3,4,5,7,8,9,10,11,6)]

# Create CSV of feature set
write.csv(features, "C:\\Users\\35383\\Documents\\Master\\Thesis\\data\\features.csv", row.names=FALSE)

write.csv(features, "C:\\Users\\35383\\Documents\\Master\\Thesis\\data\\features_nlp2.csv", row.names=FALSE)
