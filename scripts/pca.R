setwd("~/Mark work /Code")

chr1data = read.table("msmc_input_chr1.txt",as.is=T)
align = chr1data[,4]


mymat = matrix(nrow=67049,ncol=46)
intvec = integer(46)

for(j in 1:67049){
	str1 = unlist(strsplit(align[j],split=""))
	tb1 = table(str1)
	majtype = names(which(tb1 == max(tb1)))
	#note: this will give vector of length 2 for SNPs with equal frequency
	count = 1
	if(length(majtype) > 1) majtype = majtype[1] #when equal freqs
	for(k in seq(1,92,by=2)){
		if(str1[k] == majtype && str1[k+1] == majtype){
			intvec[count] = 0
		}else if(str1[k] != str1[k+1]){
			intvec[count] = 1
		}else{
			intvec[count] = 2
		}
		count = count + 1
	}
	mymat[j,] = intvec
}

forpca = t(mymat)
sdval = apply(forpca,2,"sd")
wt = sdval > 0.25
res1 = prcomp(forpca[,wt],scale=T)
cvec = c(rep(1,16),rep(2,10),rep(3,6),rep(4,14)) #from the ordering given by students
plot(res1$x[,1],res1$x[,2],xlab="PC 1", ylab= "PC 2",pch=16,col=cvec)
legend(x="topright",pch=16,col=c(1,2,3,4),legend=c("Scottish Wild","Scottish Captive","Domestic","Continent Wild"))
dev.copy2pdf(file="chr1_pca.pdf")
