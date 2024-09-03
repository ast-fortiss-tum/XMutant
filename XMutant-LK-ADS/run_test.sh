for seed in {1..20}
do
  echo "Run python script from bash, ID:$seed mode:$"
  python main_comparison_test.py --seed $((seed)) \
                                 --mutation-type RANDOM \
                                 --num-control-nodes 12 \
                                 --mutation-method random
  sleep 10                       
  python main_comparison_test.py --seed $((seed)) \
                                 --mutation-type XAI \
                                 --num-control-nodes 12 \
                                 --mutation-method random
  sleep 10

  python main_comparison_test.py --seed $((seed)) \
                                 --mutation-type XAI \
                                 --num-control-nodes 12 \
                                 --mutation-method attention_same
  sleep 10

  python main_comparison_test.py --seed $((seed)) \
                                 --mutation-type XAI \
                                 --num-control-nodes 12 \
                                 --mutation-method attention_opposite
  sleep 10

done

