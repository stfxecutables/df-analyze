# Debian (Stable)

## Build Steps

Make sure to `cd` to this directory first (although scripts will run correctly regardless
of your working directory).

I recommend to first build the sandbox so that if Python install errors occur, you
can activate the sandbox shell in writable mode with

```bash
sudo apptainer shell --no-home --writable debian_app
```

and attempt to find the missing libs or otherwise get the install working. Once working,
built wheels can also be stolen from the container and saved so they don't have to be
rebuilt on next build with

If in fact you are satisfied with that sandbox, you can convert it using

```bash
bash convert_sandbox_to_container.sh
```

Otherwise, the container is built directly using

```bash
./build_container.sh
```

