from __future__ import annotations

from enum import StrEnum

from fabrid.domain.identifiers import ClientId
from fabrid.domain.population import ClientPopulation


class NbaiotDevice(StrEnum):
    DANMINI_DOORBELL = "Danmini_Doorbell"
    ENNIO_DOORBELL = "Ennio_Doorbell"
    ECOBEE_THERMOSTAT = "Ecobee_Thermostat"
    PHILIPS_BABY_MONITOR = "Philips_B120N10_Baby_Monitor"
    PROVISION_PT_737E_CAMERA = "Provision_PT_737E_Security_Camera"
    PROVISION_PT_838_CAMERA = "Provision_PT_838_Security_Camera"
    SIMPLEHOME_XCS7_1002_CAMERA = "SimpleHome_XCS7_1002_WHT_Security_Camera"
    SIMPLEHOME_XCS7_1003_CAMERA = "SimpleHome_XCS7_1003_WHT_Security_Camera"
    SAMSUNG_SNH_1011_WEBCAM = "Samsung_SNH_1011_N_Webcam"


NBAIOT_PRIMARY_POPULATION = ClientPopulation(
    tuple(ClientId(device.value) for device in NbaiotDevice)
)

NBAIOT_DUAL_BOTNET_FAMILY_POPULATION = ClientPopulation(
    tuple(
        ClientId(device.value)
        for device in NbaiotDevice
        if device
        not in {
            NbaiotDevice.ENNIO_DOORBELL,
            NbaiotDevice.SAMSUNG_SNH_1011_WEBCAM,
        }
    )
)
