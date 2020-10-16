PURPOSE:
This repository was created to house information and reports related to the predictive modeling in ST 558 - Project 2. 

PACKAGES:
The following packages were loaded and utilized throughout this project:
caret
GGally
readr
dplyr
ggplot2
tree
bst
plyr
gbm
knitr

ANALYSIS:
To access the analyses for each day of the week, following the links below:
[Monday](Project2.md)
[Tuesday](Tuesday.md)
[Wednesday](Wednesday.md)
[Monday](Project2.md)
[Monday](Project2.md)
[Monday](Project2.md)
[Monday](Project2.md)

AUTOMATION CODE:
The following code was attempted for automating the reports for each weekend. The corresponding YAML header included params: day: and -- to access the parameters -- my code (initially) included filter(days==params$day). Unfortunately, I was unable to get the automation code to work, so I ended up hardcoding the weekday for the filters in order to generate each of the reports. 

library(markdown)

weekdays <- unique(news_data$days)

output_file <- paste0(weekdays, "Analysis.html")

params <- lapply(weekdays, FUN = function(x){list(day=x)})

reports <- tibble(output_file, params)

apply(reports, MARGIN = 1, FUN = function(x){render(input="Project2.Rmd",
                                                    output_file = x[[1]], params = x[[2]])})
