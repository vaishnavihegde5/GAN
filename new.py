import tensorflow as tf
from tensorflow.keras import layers, models

IMG_HEIGHT, IMG_WIDTH = 256, 256
LAMBDA, EPOCHS, BATCH_SIZE = 100, 50, 4

# --- Data loader: raw images + random mask ---
def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    return (tf.cast(img, tf.float32) / 127.5) - 1

def make_pair(path):
    img = load_img(path)
    mask = tf.ones([IMG_HEIGHT, IMG_WIDTH, 3])
    x = tf.random.uniform([], 64, 192, tf.int32)
    y = tf.random.uniform([], 64, 192, tf.int32)
    w = tf.random.uniform([], 32, 96, tf.int32)
    h = tf.random.uniform([], 32, 96, tf.int32)
    mask = tf.tensor_scatter_nd_update(mask, [[y, x]], [tf.zeros([h, w, 3])])
    return img * mask, img  # masked input, target

def make_ds(folder):
    files = tf.data.Dataset.list_files(folder + "/*.jpg")
    return files.map(make_pair).shuffle(100).batch(BATCH_SIZE)

# --- Down/Up blocks ---
def down(filters, size, bn=True):
    seq = models.Sequential([layers.Conv2D(filters, size, 2, 'same', use_bias=False)])
    if bn: seq.add(layers.BatchNormalization())
    seq.add(layers.LeakyReLU())
    return seq

def up(filters, size, drop=False):
    seq = models.Sequential([layers.Conv2DTranspose(filters, size, 2, 'same', use_bias=False),
                             layers.BatchNormalization()])
    if drop: seq.add(layers.Dropout(0.5))
    seq.add(layers.ReLU())
    return seq

# --- Generator (U-Net) ---
def build_gen():
    inp = layers.Input([IMG_HEIGHT, IMG_WIDTH, 3])
    downs = [down(64,4,False), down(128,4), down(256,4), down(512,4),
             down(512,4), down(512,4), down(512,4), down(512,4)]
    ups   = [up(512,4,True), up(512,4,True), up(512,4,True),
             up(512,4), up(256,4), up(128,4), up(64,4)]
    x = inp; skips=[]
    for d in downs: x=d(x); skips.append(x)
    skips=skips[:-1][::-1]
    for u,s in zip(ups,skips): x=u(x); x=layers.Concatenate()([x,s])
    out = layers.Conv2DTranspose(3,4,2,'same',activation='tanh')(x)
    return models.Model(inp,out)

# --- Discriminator (PatchGAN) ---
def build_disc():
    inp, tar = layers.Input([IMG_HEIGHT,IMG_WIDTH,3]), layers.Input([IMG_HEIGHT,IMG_WIDTH,3])
    x = layers.Concatenate()([inp, tar])
    x = down(64,4,False)(x); x = down(128,4)(x); x = down(256,4)(x)
    x = layers.Conv2D(512,4,1,'same',use_bias=False)(x)
    x = layers.LeakyReLU()(x)
    out = layers.Conv2D(1,4,1,'same')(x)
    return models.Model([inp,tar], out)

# --- Losses & optimizers ---
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
gen, disc = build_gen(), build_disc()
g_opt = tf.keras.optimizers.Adam(2e-4,0.5); d_opt = tf.keras.optimizers.Adam(2e-4,0.5)

def g_loss(d_out, fake, real):
    return bce(tf.ones_like(d_out), d_out) + LAMBDA*tf.reduce_mean(tf.abs(real-fake))

def d_loss(real_out, fake_out):
    return bce(tf.ones_like(real_out), real_out) + bce(tf.zeros_like(fake_out), fake_out)

# --- Training step ---
@tf.function
def train_step(masked, real):
    with tf.GradientTape() as gt, tf.GradientTape() as dt:
        fake = gen(masked, training=True)
        r_out, f_out = disc([masked,real],True), disc([masked,fake],True)
        gl, dl = g_loss(f_out,fake,real), d_loss(r_out,f_out)
    g_opt.apply_gradients(zip(gt.gradient(gl, gen.trainable_variables), gen.trainable_variables))
    d_opt.apply_gradients(zip(dt.gradient(dl, disc.trainable_variables), disc.trainable_variables))
    return gl, dl

# --- Training loop ---
def train(ds, epochs):
    for e in range(epochs):
        for masked, real in ds:
            gl, dl = train_step(masked, real)
        print(f"Epoch {e+1}, GenLoss={gl:.3f}, DiscLoss={dl:.3f}")