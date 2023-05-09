library("dplyr")
library("tidyr")

# Loading your example. Row names should get their own column (here `y`).
hm <- readr::read_delim("y a b c d
a 78 35 0 23
b 0 55 0 0
c 12 0 90 0
d 0 0 0 67", delim=" ")

# Gathering columns a to d
hm <- hm %>% gather(x, value, a:d)

library("ggplot2")
ggplot(hm, aes(x=x, y=y, fill=value)) + geom_tile()

ggplot(hm, aes(x=x, y=y, fill=value)) +
  geom_tile() + theme_bw() + coord_equal() +
  scale_fill_distiller(palette="Bl", direction=1) 
# Other valid palettes: Reds, Blues, Spectral, RdYlBu (red-yellow-blue), ...

ggplot(hm, aes(x=x, y=y, fill=value)) +
  geom_tile() + theme_bw() + coord_equal() +
  scale_fill_distiller(palette="RdPu", direction=1) +
  guides(fill=F) + # removing legend for `fill`
  labs(title = "Value distribution") + # using a title instead
  geom_text(aes(label=value), color="black", cexRow=10) # printing values
