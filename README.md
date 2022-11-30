# Solution for feedback-prize-3 competition  
## competition link  
https://www.kaggle.com/competitions/feedback-prize-english-language-learning  


## solution link  
https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369540  

## summray  
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5017202%2Fa680c252509a2029174d7800c59a0389%2Fsolution_summary.png?generation=1669809477192815&alt=media)  



## train    
```
python base_exp/train.py config/PATH_TO/CONFIG_FILE.yml
```

## MLM  
```
python base_exp/mlm.py config/mlm/CONFIG_FILE.yml
```

## train w/ KD  
```
python kd/train.py config/PATH_TO/CONFIG_FILE.yml
```
