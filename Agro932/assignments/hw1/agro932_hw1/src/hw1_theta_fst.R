########################
# Agro - 932 Homework 1#
# Mason Lien           #
########################


setwd("C:/Users/u942451/OneDrive - University of Nebraska-Lincoln/school/PhD/PhD_github/Agro932/assignments/hw1/agro932_hw1")

#load Geno.txt file

geno <- read.table("data/arabidopsis/mutation_01/geno.txt", header=FALSE)
names(geno) <- c("chr", "pos", "ref", "alt", "l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8", "l9", "l10", "l11", "l12", "l13", "l14", "l15", "l16", "l17", "l18", "l19", "l20")
head(geno)

#Calculate the Fst value for each site and visualize the results

for(i in 5:24){
  # replace slash and everything after it as nothing
  geno$newcol <- gsub("/.*", "", geno[,i] )
  # extract the line name
  nm <- names(geno)[i]
  # assign name for this allele
  names(geno)[ncol(geno)] <- paste0(nm, sep="_a1")
  geno$newcol <- gsub(".*/", "", geno[,i] )
  names(geno)[ncol(geno)] <- paste0(nm, sep="_a2")
}

#calculate Fst value for each site and visualize the results
#compute p1, p2, p with a 60/40 

geno$p <- apply(geno[, 25:64], 1, function(x) {sum(as.numeric(as.character(x)))})
geno$p <- geno$p/40
geno$p1 <- apply(geno[, 25:54], 1, function(x) {sum(as.numeric(as.character(x)))})
geno$p1 <- geno$p1/30
geno$p2 <- apply(geno[, 55:64], 1, function(x) {sum(as.numeric(as.character(x)))})
geno$p2 <- geno$p2/10


#calculate Fst
geno$fst <- with(geno, ((p1-p)^2 + (p2-p)^2)/(2*p*(1-p)))

#write Fst results
write.table(geno, file = "cache/fst_results_wmutation01.csv", sep = ",", row.names = F, quote = F)

#visualize Fst results with mutation rate of 0.01
fst <- read.csv("cache/fst_results_wmutation01.csv")

pdf(file = "graphs/Fst_wmutation01.PDF",   # The directory you want to save the file in
    width = 4, # The width of the plot in inches
    height = 4) # The height of the plot in inches
plot(fst$pos, fst$fst, xlab="Physical position", ylab="Fst value", main="")
dev.off()


###run same for mutation rate of 10%

geno <- read.table("data/arabidopsis/mutation_10/geno.txt", header=FALSE)
names(geno) <- c("chr", "pos", "ref", "alt", "l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8", "l9", "l10", "l11", "l12", "l13", "l14", "l15", "l16", "l17", "l18", "l19", "l20")
head(geno)

#Calculate the Fst value for each site and visualize the results

for(i in 5:24){
  # replace slash and everything after it as nothing
  geno$newcol <- gsub("/.*", "", geno[,i] )
  # extract the line name
  nm <- names(geno)[i]
  # assign name for this allele
  names(geno)[ncol(geno)] <- paste0(nm, sep="_a1")
  geno$newcol <- gsub(".*/", "", geno[,i] )
  names(geno)[ncol(geno)] <- paste0(nm, sep="_a2")
}

#calculate Fst value for each site and visualize the results
#compute p1, p2, p with a 60/40 

geno$p <- apply(geno[, 25:64], 1, function(x) {sum(as.numeric(as.character(x)))})
geno$p <- geno$p/40
geno$p1 <- apply(geno[, 25:54], 1, function(x) {sum(as.numeric(as.character(x)))})
geno$p1 <- geno$p1/30
geno$p2 <- apply(geno[, 55:64], 1, function(x) {sum(as.numeric(as.character(x)))})
geno$p2 <- geno$p2/10


#calculate Fst
geno$fst <- with(geno, ((p1-p)^2 + (p2-p)^2)/(2*p*(1-p)))

#write Fst results
write.table(geno, file = "cache/fst_results_wmutation10.csv", sep = ",", row.names = F, quote = F)

#visualize Fst results
fst <- read.csv("cache/fst_results_wmutation10.csv")

pdf(file = "graphs/Fst_wmutation10.PDF",   # The directory you want to save the file in
    width = 4, # The width of the plot in inches
    height = 4) # The height of the plot in inches
plot(fst$pos, fst$fst, xlab="Physical position", ylab="Fst value", main="")
dev.off()





