import { Text, Link, Center } from "@chakra-ui/react";
import NextLink from "next/link";

const Footer = () => {
  return (
    <Center mt="10">
      <Text>
        Made with ❤️ by{" "}
        <Link as={NextLink} color="teal.500" href="https://github.com/fani-lab">
          Fani's Lab
        </Link>
      </Text>
    </Center>
  );
};

export default Footer;
