# Output quality comparison (same 10 prompts, temperature 0)

| setting | auto-checks passed | mean tok/s | memory (Ollama /api/ps) |
|---|---|---|---|
| `qwen2.5:1.5b-instruct-fp16` | 6/7 | 62.2 | 3.22 GB |
| `qwen2.5:1.5b-instruct-q4_K_M` | 7/7 | 135.5 | 1.11 GB |

**`qwen2.5:1.5b-instruct-q4_K_M` vs `qwen2.5:1.5b-instruct-fp16`:** 6/10 identical outputs, mean text similarity 0.76

## Prompt 1
> What is 17 * 23? Answer with just the number.

**qwen2.5:1.5b-instruct-fp16** (check passed)
```
391
```

**qwen2.5:1.5b-instruct-q4_K_M** (check passed)
```
391
```

**My observation:** _(write this yourself after reading both outputs)_

## Prompt 2
> What is the capital of Australia? Answer in one word.

**qwen2.5:1.5b-instruct-fp16** (check passed)
```
Canberra
```

**qwen2.5:1.5b-instruct-q4_K_M** (check passed)
```
Canberra
```

**My observation:** _(write this yourself after reading both outputs)_

## Prompt 3
> Write a Python function is_prime(n) that returns True if n is a prime number.

**qwen2.5:1.5b-instruct-fp16** (check passed)
```
Certainly! Here's a Python function that checks if a given number `n` is a prime number:

```python
def is_prime(n):
    """
    Check if a number n is a prime number.
    
    Parameters:
    n (int): The number to check.
    
    Returns:
    bool: True if n is a prime number, False otherwise.
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# Example usage:
print(is_prime(29))  # Output: True
print(is_prime(10))  # Output: False
``
```

**qwen2.5:1.5b-instruct-q4_K_M** (check passed)
```
Certainly! Below is a Python function named `is_prime(n)` that checks whether a given number `n` is a prime number. A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.

```python
def is_prime(n):
    """Check if a number is prime."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# Example usage:
print(is_prime(6))   # False
print(is_prime(7))   # True
print(is_prime(10))  # False
print(is_prime(
```

**My observation:** _(write this yourself after reading both outputs)_

## Prompt 4
> Translate to French: 'Where is the railway station?'

**qwen2.5:1.5b-instruct-fp16** (check passed)
```
Où est la gare?
```

**qwen2.5:1.5b-instruct-q4_K_M** (check passed)
```
Où est la gare?
```

**My observation:** _(write this yourself after reading both outputs)_

## Prompt 5
> Summarize in one sentence: The Moon orbits the Earth roughly every 27 days, and its gravity causes the ocean tides. Because the Moon is slowly moving away from Earth, days are gradually getting longer.

**qwen2.5:1.5b-instruct-fp16**
```
The Moon's gravitational pull causes ocean tides, and its gradual movement away from Earth is causing days to lengthen over time.
```

**qwen2.5:1.5b-instruct-q4_K_M**
```
The Moon's gravitational pull causes ocean tides, and its gradual movement away from Earth is causing days to lengthen over time.
```

**My observation:** _(write this yourself after reading both outputs)_

## Prompt 6
> Return a JSON object with the keys "name" and "age" for a person called Asha who is 30. Output only the JSON.

**qwen2.5:1.5b-instruct-fp16** (check passed)
```
```json
{
  "name": "Asha",
  "age": 30
}
```
```

**qwen2.5:1.5b-instruct-q4_K_M** (check passed)
```
```json
{
  "name": "Asha",
  "age": 30
}
```
```

**My observation:** _(write this yourself after reading both outputs)_

## Prompt 7
> List three differences between TCP and UDP.

**qwen2.5:1.5b-instruct-fp16**
```
TCP (Transmission Control Protocol) and UDP (User Datagram Protocol) are two different protocols used for transmitting data over a network. Here are three main differences between them:

1. Connection-oriented vs Connectionless: TCP is a connection-oriented protocol, which means it establishes a connection between the sender and receiver before transmitting data. UDP, on the other hand, is a connectionless protocol, which means it does not establish a connection before transmitting data.

2. Error detection and correction: TCP includes error detection and correction mechanisms, such as checksums and retransmission mechanisms, to ensure that data is transmitted correctly. UDP, on the other hand, does not include these mechanisms, which means that it is more susceptible to errors and data loss.

3. Transmission speed: TCP is designed to provide reliable, ordered, and error-free data transmission, which means it can be slower than UDP in terms of transmission speed. UDP, on the other hand, is designed to provide fast, unreliable, and connectionless data
```

**qwen2.5:1.5b-instruct-q4_K_M**
```
TCP (Transmission Control Protocol) and UDP (User Datagram Protocol) are two fundamental protocols in the Internet Protocol Suite, each serving different purposes and providing different levels of service. Here are three key differences between TCP and UDP:

1. **Connection-oriented vs. Connectionless:**
   - **TCP** is connection-oriented, meaning it establishes a full-duplex connection between the sender and receiver before any data is sent. This connection is maintained throughout the communication, ensuring reliable delivery of data and error correction.
   - **UDP** is connectionless, meaning it does not establish a connection before sending data. It simply sends packets of data without any guarantees of delivery or error correction. This makes it more efficient for applications that can tolerate some loss or delay, such as video streaming or online gaming.

2. **Error Handling:**
   - **TCP** provides reliable, error-checked delivery of data. If a packet is lost or corrupted, TCP can retransmit the missing data or correct the error
```

**My observation:** _(write this yourself after reading both outputs)_

## Prompt 8
> A train travels 60 km in 45 minutes. What is its average speed in km/h? Show brief working.

**qwen2.5:1.5b-instruct-fp16** (check FAILED)
```
To find the average speed of the train, we use the formula for average speed, which is:

\[ \text{Average Speed} = \frac{\text{Total Distance}}{\text{Total Time}} \]

Given:
- Total Distance = 60 km
- Total Time = 45 minutes

First, we need to convert the time from minutes to hours because the speed is asked in km/h. There are 60 minutes in an hour, so:

\[ 45 \text{ minutes} = \frac{45}{60} \text{ hours} = \frac{3}{4} \text{ hours} \]

Now, we can calculate the average speed:

\[ \text{Average Speed} = \frac{60 \text{ km}}{\frac{3}{4} \text{ hours}} \]

To divide by a fraction, we multiply by its reciprocal:

\[ \text{Average Speed} = 60
```

**qwen2.5:1.5b-instruct-q4_K_M** (check passed)
```
To find the average speed of the train, we use the formula:

\[ \text{Average Speed} = \frac{\text{Distance}}{\text{Time}} \]

Given:
- Distance = 60 km
- Time = 45 minutes

First, we need to convert the time from minutes to hours because the speed is asked in km/h. There are 60 minutes in an hour, so:

\[ 45 \text{ minutes} = \frac{45}{60} \text{ hours} = 0.75 \text{ hours} \]

Now we can calculate the average speed:

\[ \text{Average Speed} = \frac{60 \text{ km}}{0.75 \text{ hours}} \]

\[ \text{Average Speed} = 80 \text{ km/h} \]

So, the average speed of the train is 80 km/h.
```

**My observation:** _(write this yourself after reading both outputs)_

## Prompt 9
> Explain what a hash table is to a 12 year old in three sentences.

**qwen2.5:1.5b-instruct-fp16**
```
A hash table is like a special kind of dictionary where each word or number has a unique code. When you look up a word or number, the code helps quickly find the word or number you're looking for, just like how you might use a shortcut to find your favorite toy in a big room.
```

**qwen2.5:1.5b-instruct-q4_K_M**
```
A hash table is like a special kind of dictionary where you can quickly find the word you're looking for. It's super fast because it uses a special code (called a hash) to find the word, just like how you might use a shortcut to find your favorite toy in a messy room. This makes it super useful for games or when you're trying to remember lots of things quickly!
```

**My observation:** _(write this yourself after reading both outputs)_

## Prompt 10
> Who wrote the novel 'Pride and Prejudice'? Answer in one short sentence.

**qwen2.5:1.5b-instruct-fp16** (check passed)
```
Jane Austen wrote the novel 'Pride and Prejudice'.
```

**qwen2.5:1.5b-instruct-q4_K_M** (check passed)
```
Jane Austen wrote the novel 'Pride and Prejudice'.
```

**My observation:** _(write this yourself after reading both outputs)_
