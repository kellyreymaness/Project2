Project 2: Modeling & Report Automation
================
Kelly Baker
10/9/2020

  - [**INTRODUCTION**](#introduction)
  - [**DATA**](#data)
  - [**SUMMARIZATION**](#summarization)
  - [**MODELS**](#models)
      - [**Model 1: Regression Tree
        Model**](#model-1-regression-tree-model)
      - [**Model 2: Boosted Tree Model**](#model-2-boosted-tree-model)
  - [**PREDICTION**](#prediction)

# **INTRODUCTION**

In this vignette, we’ll look at various attributes of articles published
by Mashable in an effort to predict popularity. For the popularity, the
response, we’ll use number of social media shares as a proxy. The
predictors, oh which there are nine, were chosen based on their
correlation with the response. The nine independent variables are:  
\* *n\_non\_stop\_words*: rate of unique non-stop words  
\* *num\_hrefs*: number of links  
\* *data\_channel\_is\_world*: is data channel world? 1=yes, 0=no  
\* *kw\_avg\_avg*: average keywords, average shares  
\* *self\_reference\_min\_shares*: minimum shares of referenced articles
in Mashable  
\* *self\_reference\_avg\_sharess*: Average shares of referenced
articles in Mashable  
\* *LDA\_03*: Closeness to LDA topic 3  
\* *global\_subjectivity*: text subjectivity  
\* *avg\_negative\_polarity*: average polarity of negative words

# **DATA**

In the code chunk below, I’ve read in my .csv dataset,
OnlineNewsPopularity, via the read\_csv() function, and creating a new
variable, “days”, that simplifies data on which day of the week articles
were published. Using “days”, I’ve also filtered the data by day, and
selected all variables except for url and timedelta (as they are
non-predictive variables). <br> Next, I’ve split my data into two parts:
a training set and a test set using helpful function from the `caret`
package. With the createDataPartion() function, I’ve indexed 70% of the
data which I was then able to call and save into a training dataset
(train\_news). The remaining 30% of data was saved as the test data in a
new object called test\_news.

``` r
news_data <- read_csv(file = "OnlineNewsPopularity.csv")
```

    ## Parsed with column specification:
    ## cols(
    ##   .default = col_double(),
    ##   url = col_character()
    ## )

    ## See spec(...) for full column specifications.

``` r
news_data <- news_data %>% mutate(days = ifelse(news_data$weekday_is_monday == 1, "Monday",
               ifelse(news_data$weekday_is_tuesday == 1, "Tuesday", 
                      ifelse(news_data$weekday_is_wednesday==1, "Wednesday",
                             ifelse(news_data$weekday_is_thursday==1, "Thursday",
                                    ifelse(news_data$weekday_is_friday==1, "Friday",
                                           ifelse(news_data$weekday_is_saturday, "Saturday", "Sunday")))))))

news_data <- news_data %>% filter(days=="Friday") %>% select(-url, -timedelta)
news_data
```

    ## Registered S3 method overwritten by 'cli':
    ##   method     from
    ##   print.tree tree

    ## # A tibble: 5,701 x 60
    ##    n_tokens_title n_tokens_content n_unique_tokens n_non_stop_words
    ##             <dbl>            <dbl>           <dbl>            <dbl>
    ##  1             12              698           0.499             1.00
    ##  2             13              310           0.612             1.00
    ##  3              8              461           0.550             1.00
    ##  4              9              318           0.579             1.00
    ##  5             10              294           0.703             1.00
    ##  6              9              930           0.442             1.00
    ##  7             10              682           0.487             1.00
    ##  8             10              349           0.542             1.00
    ##  9              5             1302           0.414             1.00
    ## 10             13              220           0.603             1.00
    ## # ... with 5,691 more rows, and 56 more variables:
    ## #   n_non_stop_unique_tokens <dbl>, num_hrefs <dbl>, num_self_hrefs <dbl>,
    ## #   num_imgs <dbl>, num_videos <dbl>, average_token_length <dbl>,
    ## #   num_keywords <dbl>, data_channel_is_lifestyle <dbl>,
    ## #   data_channel_is_entertainment <dbl>, data_channel_is_bus <dbl>,
    ## #   data_channel_is_socmed <dbl>, data_channel_is_tech <dbl>,
    ## #   data_channel_is_world <dbl>, kw_min_min <dbl>, kw_max_min <dbl>,
    ## #   kw_avg_min <dbl>, kw_min_max <dbl>, kw_max_max <dbl>, kw_avg_max <dbl>,
    ## #   kw_min_avg <dbl>, kw_max_avg <dbl>, kw_avg_avg <dbl>,
    ## #   self_reference_min_shares <dbl>, self_reference_max_shares <dbl>,
    ## #   self_reference_avg_sharess <dbl>, weekday_is_monday <dbl>,
    ## #   weekday_is_tuesday <dbl>, weekday_is_wednesday <dbl>,
    ## #   weekday_is_thursday <dbl>, weekday_is_friday <dbl>,
    ## #   weekday_is_saturday <dbl>, weekday_is_sunday <dbl>, is_weekend <dbl>,
    ## #   LDA_00 <dbl>, LDA_01 <dbl>, LDA_02 <dbl>, LDA_03 <dbl>, LDA_04 <dbl>,
    ## #   global_subjectivity <dbl>, global_sentiment_polarity <dbl>,
    ## #   global_rate_positive_words <dbl>, global_rate_negative_words <dbl>,
    ## #   rate_positive_words <dbl>, rate_negative_words <dbl>,
    ## #   avg_positive_polarity <dbl>, min_positive_polarity <dbl>,
    ## #   max_positive_polarity <dbl>, avg_negative_polarity <dbl>,
    ## #   min_negative_polarity <dbl>, max_negative_polarity <dbl>,
    ## #   title_subjectivity <dbl>, title_sentiment_polarity <dbl>,
    ## #   abs_title_subjectivity <dbl>, abs_title_sentiment_polarity <dbl>,
    ## #   shares <dbl>, days <chr>

``` r
set.seed(50)
index_train <- createDataPartition(news_data$shares, p=0.7, list=FALSE)
train_news <- news_data[index_train, ]
```

    ## Warning: The `i` argument of ``[`()` can't be a matrix as of tibble 3.0.0.
    ## Convert to a vector.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_warnings()` to see where this warning was generated.

``` r
test_news <- news_data[-index_train, ]
```

# **SUMMARIZATION**

The following code generates a slew of diagnostic plots. The response
was grouped with several predictors in order to assess scatter and
correlation.

``` r
ggpairs(train_news %>% select(n_tokens_title:n_non_stop_words, shares), title="Group 1 Diagnostics")
```

![](Friday_files/figure-gfm/plot1-1.png)<!-- -->

``` r
ggpairs(train_news %>% select(n_non_stop_unique_tokens:num_imgs, shares), title="Group 2 Diagnostics")
```

![](Friday_files/figure-gfm/plot1-2.png)<!-- -->

``` r
ggpairs(train_news %>% select(num_videos:data_channel_is_lifestyle, shares), title="Group 3 Diagnostics")
```

![](Friday_files/figure-gfm/plot1-3.png)<!-- -->

``` r
ggpairs(train_news %>% select(data_channel_is_entertainment:data_channel_is_tech, shares), title="Group 4 Diagnostics")
```

![](Friday_files/figure-gfm/plot1-4.png)<!-- -->

``` r
ggpairs(train_news %>% select(data_channel_is_world:kw_avg_min, shares), title="Group 5 Diagnostics")
```

![](Friday_files/figure-gfm/plot1-5.png)<!-- -->

``` r
ggpairs(train_news %>% select(kw_min_max:kw_min_avg, shares), title="Group 6 Diagnostics")
```

![](Friday_files/figure-gfm/plot1-6.png)<!-- -->

``` r
ggpairs(train_news %>% select(kw_max_avg:self_reference_max_shares, shares), title="Group 7 Diagnostics")
```

![](Friday_files/figure-gfm/plot1-7.png)<!-- -->

``` r
ggpairs(train_news %>% select(self_reference_avg_sharess, LDA_00:LDA_02, shares), title="Group 8 Diagnostics")
```

![](Friday_files/figure-gfm/plot1-8.png)<!-- -->

``` r
ggpairs(train_news %>% select(LDA_03:global_sentiment_polarity, shares), title="Group 9 Diagnostics")
```

![](Friday_files/figure-gfm/plot1-9.png)<!-- -->

``` r
ggpairs(train_news %>% select(global_rate_positive_words:rate_negative_words, shares), title="Group 10 Diagnostics")
```

![](Friday_files/figure-gfm/plot1-10.png)<!-- -->

``` r
ggpairs(train_news %>% select(avg_positive_polarity:avg_negative_polarity, shares), title = "Group 11 Diagnostics")
```

![](Friday_files/figure-gfm/plot1-11.png)<!-- -->

``` r
ggpairs(train_news %>% select(min_negative_polarity:title_sentiment_polarity, shares), title="Group 12 Diagnostics")
```

![](Friday_files/figure-gfm/plot1-12.png)<!-- -->

``` r
ggpairs(train_news %>% select(abs_title_subjectivity:shares), title="Group 13 Diagnostics")
```

![](Friday_files/figure-gfm/plot1-13.png)<!-- -->

# **MODELS**

The code chunks below fit two potential models on the training data: a
regression tree (non-ensemble) and a boosted tree (ensemble).

## **Model 1: Regression Tree Model**

Using the train() function and various options, the optimal regression
tree model is fit on the training data using leave-one-out cross
validation and a tuning parameter, “cp”. Note: method=“rpart” is what
tells R to fit a regression tree.

``` r
tree_fit <- train(shares ~ n_non_stop_words + num_hrefs + data_channel_is_world + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + LDA_03 + global_subjectivity + avg_negative_polarity, data=train_news,
                  method="rpart",
                  trControl = trainControl(method="LOOCV", number=3),
                  metric="MAE",
                  preProcess = c("center", "scale"),
                  tuneGrid = expand.grid(cp = seq(0, 0.2, .02)))

tree_fit$bestTune
```

    ##     cp
    ## 2 0.02

## **Model 2: Boosted Tree Model**

The second model was also fit using the train() function. Method=“gbm”
was used to instruct R to fit a boosted tree. Also, 4 tuning parameters
were used. The values for the tuning parameters shrinkage, n.trees, and
interaction.depth were chosen through cross validation to minimize
prediction error. (Note: the fourth tuning parameter, n.minobsinnode,
was set to the default value of 10).

``` r
boost_fit <- train(shares ~ n_non_stop_words + num_hrefs + data_channel_is_world + kw_avg_avg + self_reference_min_shares +
                     self_reference_avg_sharess + LDA_03 + global_subjectivity + avg_negative_polarity, data=train_news, 
                   method="gbm", 
                   trControl = trainControl(method="cv", number=3), 
                   preProcess = c("center", "scale"),
                   tuneGrid = expand.grid(n.trees = seq(100, 1000, 25), interaction.depth = seq(1, 9, 1), 
                                          shrinkage = seq(0, 0.2, .02), n.minobsinnode=10))
```

    ## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo, :
    ## There were missing values in resampled performance measures.

``` r
boost_fit$bestTune
```

# **PREDICTION**

In the code chunk below, I’m using the predict() function to fit my
selected models to the test data. The first code group uses the fit from
the regression tree model, while the second code group uses the fit from
the boosted tree model. Through this process, we’re given information
about prediction error, which we want to minimize. The code calls
`pred_tree_gfm` and `pred_boost_gfm` reveals three metrics(RMSE,
Rsquared, and MAE) for understanding error.

``` r
pred_tree <- predict(tree_fit, newdata = test_news)
pred_tree_gfm <- postResample(pred_tree, test_news$shares)

pred_boost <- predict(boost_fit, newdata = test_news)
pred_boost_gfm <- postResample(pred_boost, test_news$shares)


pred_tree_gfm
```

    ##         RMSE     Rsquared          MAE 
    ## 5.616114e+03 2.946771e-03 2.757731e+03

``` r
pred_boost_gfm
```

    ##         RMSE     Rsquared          MAE 
    ## 5.114558e+03 4.637494e-02 2.605658e+03
