## Contributors
Ryan Campbell and Jiachen Ding

## Example Usage

```python
from Markov import CountMinMarkov

# Initialize model
cmm = CountMinMarkov(num_hash=25, length_table=2000)
# Update model based on text files
cmm.parse_text('example.txt', encoding='utf8')
# Generate new text similar to that in 'example.txt'
print(cmm.get_sentence())
```

