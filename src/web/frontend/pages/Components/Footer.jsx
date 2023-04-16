import { Text, Link } from "@chakra-ui/react";
import NextLink from "next/link";

const Footer = () => {
  return (
    <Text>
      Made with ❤️ by{" "}
      <Link as={NextLink} color="teal.500" href="https://github.com/fani-lab">
        Fani's Lab
      </Link>
    </Text>
  );
};

export default Footer;
