<div align="center">
	<h1>Physarum Biocomputer</h1>
</div>

<hr/>

![](./assets/biocomputer-render-1.png)

<hr/>

## Project Structure

#### Descriptions

The project structure is organized as follows:

- 📂 `adaptors`: contains all of the hardware files for the various adaptors that are compatible with the biocomputer carrier board.

- 📂 `data`: contains all of the experimentally collected data.

- 📂 `devices`: contains all of the hardware files for both the open-air and enclosure-type devices.

- 📂 `modules`: contains all of the hardware files for each of the biocomputer modules.

- 📂 `software`: contains all of the software used in performing experiments.

#### Structure
📂 `repository` 

- 📂 `data`
	- 📂 `device-pulse-recordings` 
	- 📂 `device-regression-recordings` 
	- 📂 `device-sweep-recordings` 
	- 📂 `eis-recordings`
	- 📂 `noise-recordings`
	- 📂 `reservoir-computing-recordings`
- 📂 `modules`
    - 📂 `biocomputer-eval-motherboard`
    - 📂 `dac-eval-module`
    - 📂 `daq-eval-board`
    - 📂 `dual-rail-power-eval-module`
    - 📂 `selector-eval-module`
- 📂 `adaptors`
    - 📂 `1x1-adaptor`
    - 📂 `4x1-adaptor`
	- 📂 `8x1-adaptor`
    - 📂 `card-half`
    - 📂 `card-full`
    - 📂 `configurable-adaptor`
    - 📂 `standard-adaptor`
- 📂 `software`
    - 📂 `examples`
- 📂 `devices`
    - 📂 `enclosures`
    - 📂 `pluggable`
