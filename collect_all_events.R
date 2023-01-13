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

common_cols <- intersect(colnames(KeyEvents18), colnames(KeyEvents19))
KeyEvents18 <- KeyEvents18[, common_cols]
KeyEvents19 <- KeyEvents19[, common_cols]

merge <- rbind(KeyEvents18, KeyEvents19)

common_cols2 <- intersect(colnames(merge), colnames(KeyEvents21))
KeyEvents21 <- KeyEvents21[, common_cols2]
merge <- merge[, common_cols2]
merge2 <- rbind(merge, KeyEvents21)

Data <- merge2[, c("index", "timestamp", "possession", "possession_team.name", "duration", 
                   "type.name", "pass.length", "pass.switch", 
                   "pass.cross", "pass.through_ball", "pass.height.name",
                   "location.x", "location.y", "carry.end_location.x", "carry.end_location.y",
                   "pass.end_location.x", "pass.end_location.y")]

write.csv(Data, "C:\\Users\\35383\\Documents\\Master\\Thesis\\data\\allevents.csv", row.names=FALSE)