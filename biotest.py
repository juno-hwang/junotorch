!pip install Biopython

import Bio

from Bio.Seq import Seq

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

seq1=Seq("ATATATACCCCGGGGGGG")

type(seq1)

for index, letter in enumerate(seq1):
    print(index,letter)

print(seq1[4])

seq1[0:15]

seq1[0::1]

seq1[::-1]

countG=seq1.count("G")
countC=seq1.count("C")
countGC=(countG+countC)/len(seq1)*100
print(countGC)


from Bio.SeqUtils import GC
GC(seq1)

seq1.complement()

seq1.reverse_complement()

from Bio import SeqIO
from Bio import Entrez

Entrez.email="자신의 이메일"


handle=Entrez.efetch(db="nucleotide", rettype="fasta",id="MN996528.1")
for seq_rec1 in SeqIO.parse(handle,"fasta"):
    count=SeqIO.write(seq_rec1,"sars-cov.fasta","fasta")

print(seq_rec1)

seq2=seq_rec1.seq
print(seq2)

seq2.complement()

seqrev2=seq2.reverse_complement()
print(seqrev2)


from Bio.Seq import Seq

# DNA 시퀀스


start_codon = "ATG"

stop_codons = ["TAA", "TAG", "TGA"]

start_position = seq2.find(start_codon)

stop_position = min([seq2.find(stop_codon, start_position) for stop_codon in stop_codons if seq2.find(stop_codon, start_position) != -1])

extracted_sequence = seq2[start_position:stop_position+3]

print("Extracted Sequence:", extracted_sequence)
