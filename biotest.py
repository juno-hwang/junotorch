from Bio import Entrez
from Bio import SeqIO

def efetch(db, rettype, id):
    import io, urllib, ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db={}&rettype={}&id={}&tool=biopython".format(db, rettype, id)
    handle =  urllib.request.urlopen(url)
    handle =  io.TextIOWrapper(handle, encoding='utf-8')
    return handle

handle = efetch(db="nucleotide", rettype="fasta", id="MN996528.1")
for seq_rec1 in SeqIO.parse(handle,"fasta"):
    count=SeqIO.write(seq_rec1,"sars-cov.fasta","fasta")

print(seq_rec1)

seq2=seq_rec1.seq
print(seq2)

seq2.complement()

seqrev2=seq2.reverse_complement()
print(seqrev2)
