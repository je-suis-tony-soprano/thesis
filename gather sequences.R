library(tidyverse)
library(StatsBombR)

Comp1 <- FreeCompetitions() %>%
  filter(competition_id==11 & season_id==1)

Comp2 <- FreeCompetitions() %>%
  filter(competition_id==11 & season_id==4)

Comp3 <- FreeCompetitions() %>%
  filter(competition_id==11 & season_id==90)

Matches18 <- FreeMatches(Comp1)
Matches19 <- FreeMatches(Comp2)
Matches21 <- FreeMatches(Comp3)

StatsBombData18 <- free_allevents(MatchesDF = Matches18, Parallel = T)
StatsBombData19 <- free_allevents(MatchesDF = Matches19, Parallel = T)
StatsBombData21 <- free_allevents(MatchesDF = Matches21, Parallel = T)

Data18 <- allclean(StatsBombData18)
Data19 <- allclean(StatsBombData19)
Data21 <- allclean(StatsBombData21)

KeyEvents18 <- Data18 %>%
  filter(type.name == 'Pass' | type.name == 'Carry' | type.name == 'Dribble' | type.name == 'Shot')

KeyEvents19 <- Data19 %>%
  filter(type.name == 'Pass' | type.name == 'Carry' | type.name == 'Dribble' | type.name == 'Shot')

KeyEvents21 <- Data21 %>%
  filter(type.name == 'Pass' | type.name == 'Carry' | type.name == 'Dribble' | type.name == 'Shot')

merge <- rbind(KeyEvents18, KeyEvents19)
merge2 <- rbind(merge, KeyEvents21)

Data <- merge2[, c("index", "timestamp", "possession", "possession_team.name", "duration", 
                      "type.name", "pass.length", "pass.switch", 
                      "pass.cross", "pass.through_ball", "pass.height.name",
                      "location.x", "location.y", "carry.end_location.x", "carry.end_location.y",
                      "pass.end_location.x", "pass.end_location.y")]

Sequences <- Data %>% group_by(possession) %>%
  summarise(number_of_passes = sum(type.name == "Pass"),
            number_of_dribbles = sum(type.name == "Dribble"),
            number_of_carries = sum(type.name == "Carry"),
            attack_speed = mean(duration),
            shot = sum(type.name == "Shot"))

start <- Data %>% group_by(possession) %>%
  slice(1) %>%
  select(c(3,13,14))

end <- Data %>% group_by(possession) %>%
  slice_tail() %>%
  select(c(3, 13:18))

Sequences$start_location_x <- start$location.x
Sequences$start_location_y <- start$location.y
Sequences$end_location_x <- end$location.x
Sequences$end_location_y <- end$location.y
Sequences$carry.end_location_x <- end$carry.end_location.x
Sequences$carry.end_location_y <- end$carry.end_location.y
Sequences$pass.end_location_x <- end$pass.end_location.x
Sequences$pass.end_location_y <- end$pass.end_location.y

clean_shots <- replace(Sequences$shot, Sequences$shot>1, 1)
Sequences$shot <- clean_shots

Sequences <- Sequences %>%
  filter(start_location_x != 120)

Sequences$start_distance_x <- 120 - Sequences$start_location_x
Sequences$end_distance_x <- 120 - Sequences$end_location_x

write.csv(Data, "C:\\Users\\35383\\Documents\\Master\\Thesis\\sequences.csv", row.names=FALSE)
















