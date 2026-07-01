from collections import Counter
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from multiprocessing import Pool

class BPE:

    special_token = b"<|endoftext|>"
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    num_processes = 16
    file_name = "data/TinyStoriesV2-GPT4-train.txt"

    
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.main()


    def pretokenize(self, file: str, start: int, end: int) -> Counter[bytes]:
        pretoken_cnt = Counter[bytes]()
        pattern = f"({re.escape(self.special_token.decode("utf-8"))})"

        with open(file, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
            docs = re.split(pattern, chunk)
            
            for doc in docs:
                pretokens = re.finditer(self.PAT.encode(), doc.encode())
                for pretoken in pretokens:
                    key = pretoken.group()
                    pretoken_cnt[key] += 1
        
        return pretoken_cnt

    
    def train(self):
        pass

    
    def main(self):
        pretoken_cnt = Counter[bytes]()
        with open(self.file_name, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, self.special_token)

            for b in boundaries:
                print(b)
            
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            with Pool(processes=self.num_processes) as pool:
                results = pool.starmap(self.pretokenize, zip(
                    [self.file_name] * (len(boundaries) - 1),
                    boundaries[:-1],
                    boundaries[1:]
                    ))

            for r in results:
                pretoken_cnt.update(r)

            for k, v in sorted(pretoken_cnt.items(), key=lambda pair: pair[1], reverse=True)[:20]:
                print(repr(k), v)
        
        self.train()


if __name__ == "__main__":
    bpe = BPE("data/TinyStoriesV2-GPT4-train.txt")
    
